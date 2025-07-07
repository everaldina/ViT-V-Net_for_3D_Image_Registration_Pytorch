import os, utils
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import argparse
import pickle
from data.data_utils import pkload
import torch

def args_input():
    parser = argparse.ArgumentParser(description='ViT-V-Net')
    parser.add_argument('data_folder', type=str, help='data directory')
    parser.add_argument('transform_path', type=str, default='experiments/', help='model directory')
    parser.add_argument('--result_path', type=str, default='aligned/', help='result directory')
    parser.add_argument('--reshape_mode', type=str, default='zoom', help='reshape mode (zoom or reduce)')
    parser.add_argument('--image_type', type=str, default='normal', help='image type (normal or transformed)')
    parser.add_argument('--suffix', type=str, default='', help='suffix for saving')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--file_extension', type=str, default='nib', help='file extension')
    return parser.parse_args()


def save_out(array, path, scan_name):
    if not os.path.exists(path):
        os.makedirs(path)
        
    img = np.squeeze(array)
    
    with open(os.path.join(path, f'{scan_name}.pkl'), "wb") as f:
        pickle.dump(img, f)
        


    

def main():
    args = args_input()
    data_folder = args.data_folder
    transform_path = args.transform_path
    result_path = args.result_path
    reshape_mode = args.reshape_mode
    image_type = args.image_type
    save_suffix = args.suffix if args.suffix == '' else f'_{args.suffix}'
    num_workers = args.num_workers
    img_size = (64, 512, 512)
    extension = args.file_extension
    


    if not os.path.exists(transform_path):
        print('Model directory not found')
        return

    ids = {}
    for x in os.listdir(data_folder):
        if (x.endswith('.nii.gz') and extension == 'nib') or (x.endswith('.pkl') and extension == 'pkl'):
            name = x.split('.')[0]
            ids[name] =  {
                'path': os.path.join(data_folder, x),
                'y_shape': None,
                'flows': []
            }
            
    if len(ids) == 0:
        print('No data found')
        return
    
    for img in ids:
        folder_transform = os.path.join(transform_path, img)
        cascades = [x for x in os.listdir(folder_transform) if x.startswith('cascade')]
        order_cascades = sorted(cascades, key=lambda x: int(x.split('_')[-1]))
        for cascade in order_cascades:
            if ids[img]['y_shape'] is None:
                ids[img]['y_shape'] = pkload(os.path.join(folder_transform, cascade, 'y.pkl')).shape
            flow = os.path.join(folder_transform, cascade, 'flow.pkl')
            ids[img]['flows'].append(flow)
    
    print('Loading model...')
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    
    # flows = [os.path.join(transform_path, x, 'flow.pkl') for x in cascades]
    # imgs = [img_path] * len(flows)
    
    imgs = []
    y_shapes = []
    for img in ids:
        imgs = imgs + [ids[img]['path']] 
        y_shapes = y_shapes + [ids[img]['y_shape']]
    
    img_set = datasets.OrcaScoreData(imgs, y_shapes, img_size, image_type, reshape_mode, extension)
    img_loader = DataLoader(img_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    for data in img_loader:
        img = data['x']
        id = data['id'][0]
        print(f'---- Processing {id} ---')
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
            print(f"Flow shape: {flow.shape}")
            print(f"Flow type: {flow.dtype}")
            
            def_out = reg_model([def_outs[-1].cuda().float(), flow])
            print(f"Def_out shape: {def_out.shape}")
            print(f"Def_out type: {def_out.dtype}")
            print(f"Def_out min: {def_out.min()}")
            print(f"Def_out max: {def_out.max()}")
            print(f"Def_out sum: {def_out.sum()}")
            
            def_outs.append(def_out.cuda())
            
            save_folder = os.path.join(result_path, id)
            save_out(def_out.cpu().numpy(), save_folder, f'cascade_{i + 1}{save_suffix}')


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