import os, utils
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import argparse
import pickle
from data.data_utils import pkload
import torch
import SimpleITK as sitk
from utils import zoom_img
import pandas as pd


def get_calc_label(calc_path):
    dirs = os.listdir(calc_path)
    calcs = {}
    
    for d in dirs:
        path = os.path.join(calc_path, d)
        if os.path.isfile(path) and path.endswith('.mhd'):
            name = d.split('.')[0][:-1]
            calcs[name] = path
    return calcs
            
def num_arteries(artery_label):
    return np.sum(artery_label > 0)

def num_calc(calc_label):
    return np.sum(calc_label > 0)
        

def dice_calc(artery_label, calc_label):
    calc_unico = calc_label > 0
    intersection = np.sum(artery_label * calc_unico)
    return 2.0 * intersection / (np.sum(artery_label) + np.sum(calc_unico))

def calc_intersection(artery_label, calc_label):
    calc_unico = calc_label > 0
    intersection = np.sum(artery_label * calc_unico)
    if np.sum(calc_unico) == 0:
        intersection_percentage = 0
    else:
        intersection_percentage = intersection / np.sum(calc_unico)
    return intersection, intersection_percentage

def args_input():
    parser = argparse.ArgumentParser(description='ViT-V-Net')
    parser.add_argument('artery_path', type=str, help='data directory')
    parser.add_argument('calc_path', type=str, default='experiments/', help='model directory')
    parser.add_argument('transform_path', type=str, help='model directory')
    parser.add_argument('--result_path', type=str, default='infer_orca/', help='result directory')
    parser.add_argument('--reshape_mode', type=str, default='zoom', help='reshape mode (zoom or reduce)')
    parser.add_argument('--image_type', type=str, default='normal', help='image type (normal or transformed)')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    return parser.parse_args()


def main():
    args = args_input()
    artery_path = args.artery_path
    calc_path = args.calc_path
    transform_path = args.transform_path
    result_path = args.result_path
    reshape_mode = args.reshape_mode
    image_type = args.image_type
    num_workers = args.num_workers
    img_size = (64, 512, 512)
    


    if not os.path.exists(transform_path):
        print('Model directory not found')
        return

    ids = {}
    for x in os.listdir(artery_path):
        if (x.endswith('.nii.gz')):
            name = x.split('.')[0]
            ids[name] =  {
                'path': os.path.join(artery_path, x),
                'y_shape': None,
                'flows': []
            }
            
    if len(ids) == 0:
        print('No data found')
        return
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    for img in ids:
        folder_transform = os.path.join(transform_path, img)
        cascades = [x for x in os.listdir(folder_transform) if x.startswith('cascade')]
        order_cascades = sorted(cascades, key=lambda x: int(x.split('_')[-1]))
        for cascade in order_cascades:
            if ids[img]['y_shape'] is None:
                ids[img]['y_shape'] = pkload(os.path.join(folder_transform, cascade, 'y.pkl')).shape
            flow = os.path.join(folder_transform, cascade, 'flow.pkl')
            ids[img]['flows'].append(flow)
            
            
    calcs = get_calc_label(calc_path)
    
    print('Loading model...')
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    
    
    imgs = []
    y_shapes = []
    for img in ids:
        imgs = imgs + [ids[img]['path']] 
        y_shapes = y_shapes + [ids[img]['y_shape']]
    
    img_set = datasets.OrcaScoreData(imgs, y_shapes, img_size, image_type, reshape_mode)
    img_loader = DataLoader(img_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    
    results = {
        'scan': [],
        'num_arteries': [],
        'num_calc': [],
        'intersection': [],
        'intersection_rate': [],
        'dice': [],
        'num_cascades': []
    }
    
    
    for data in img_loader:
        img = data['x']
        id = data['id'][0]
        print(f'---- Processing {id} ---')
        print(f"Image shape: {img.shape}")
        print(f"Image type: {img.dtype}")
        
        calc = calcs[id]
        calc = sitk.ReadImage(calc)
        calc = sitk.GetArrayFromImage(calc)
        calc = zoom_img(calc, img_size)
        
        results['scan'].append(id)
        results['num_arteries'].append(num_arteries(img.detach().cpu().numpy()))
        results['num_calc'].append(num_calc(calc))
        intersection, intersection_rate = calc_intersection(img.detach().cpu().numpy(), calc)
        results['intersection'].append(intersection)
        results['intersection_rate'].append(intersection_rate)
        results['dice'].append(dice_calc(img.detach().cpu().numpy(), calc))
        results['num_cascades'].append(0)
        
        
        
        img = img.cuda()
        flows = ids[id]['flows']
        def_outs = [img]
        for i in range(len(flows)):
            
            print(f' - Processing cascade {i + 1}')
            flow = pkload(flows[i])
            # add two channels for flow
            flow = np.expand_dims(flow, 0)
            flow = np.repeat(flow, img.shape[0], axis=0)
            flow = torch.from_numpy(flow).cuda()
            
            
            def_out = reg_model([def_outs[-1].cuda().float(), flow])
            
            results['scan'].append(id)
            results['num_arteries'].append(num_arteries(def_out.detach().cpu().numpy()))
            results['num_calc'].append(num_calc(calc))
            intersection, intersection_rate = calc_intersection(def_out.detach().cpu().numpy(), calc)
            results['intersection'].append(intersection)
            results['intersection_rate'].append(intersection_rate)
            results['dice'].append(dice_calc(def_out.detach().cpu().numpy(), calc))
            results['num_cascades'].append(i + 1)
            
            
            def_outs.append(def_out.cuda())
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, 'intersection.csv'))
            


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()