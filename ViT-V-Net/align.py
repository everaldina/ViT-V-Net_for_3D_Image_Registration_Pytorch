import glob
import os, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from models import CONFIGS as CONFIGS_ViT_seg
import argparse
import pickle
from data.data_utils import pkload

def args_input():
    parser = argparse.ArgumentParser(description='ViT-V-Net')
    parser.add_argument('--test_dir', type=str, default='/vit-v-net/test/', help='testing data directory')
    parser.add_argument('--model_folder', type=str, default='vit-v-net/', help='model directory')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--cascade_number', type=int, default=1, help='number of cascades')
    
    return parser.parse_args()

def get_model(path):
    for f in os.listdir(path):
        if f.endswith('.pth.tar'):
            return os.path.join(path, f)
    return None

def save_array(array, path, scan_name):
    if not os.path.exists(path):
        os.makedirs(path)
        
    img = np.squeeze(array)
    
    with open(os.path.join(path, f'{scan_name}.pkl'), "wb") as f:
        pickle.dump(img, f)
        
def vit_pickles(path, cascade = 1):
    files_list = []
    for f in os.listdir(path):
        if not os.path.isdir(os.path.join(path, f)):
            continue
        id = f.split('.pkl')[0]
        scan = os.path.join(path, id, f'cascade_{cascade}')
        x = pkload(os.path.join(scan, 'x_to_y.pkl'))
        y = pkload(os.path.join(scan, 'y.pkl'))
        data = (x, y)
        
        with open(os.path.join(scan, f'cascade_{cascade}_results.pkl'), "wb") as pkl:
            pickle.dump(data, pkl)
        
        files_list.append(os.path.join(scan, f'cascade_{cascade}_results.pkl'))
    return files_list
        
        
@torch.no_grad()
def align(model, ids, test_loader, model_dir, cascade = 1):
    for id, data in zip(ids, test_loader):
        save_folder = os.path.join(model_dir, id, f'cascade_{cascade}')
        print(f'Processing {id}')
        
        model.eval()
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]

        print(f' - Forward pass')
        x_in = torch.cat((x,y),dim=1)
        x_def, flow = model(x_in)
        del x_in
        save_array(flow.cpu().numpy(), save_folder, f'flow')
        del flow
        
        print(f' - Backward pass')
        y_in = torch.cat((y, x), dim=1)
        y_def, flow = model(y_in)
        del y_in
        del flow
        
        print(f' - Saving images')
        x_def = x_def.cpu().numpy()
        y_def = y_def.cpu().numpy()
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        
        save_array(x_def, save_folder, f'x_to_y')
        save_array(y_def, save_folder, f'y_to_x')
        save_array(x, save_folder, f'x')
        save_array(y, save_folder, f'y')
        
    

def main():
    args = args_input()
    test_dir = args.test_dir
    model_folder = args.model_folder
    model_dir = 'experiments/' + model_folder
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    num_workers = args.num_workers
    img_size = (64, 512, 512)
    cascades = args.cascade_number


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print('Loading model...')
    model = models.ViTVNet(config_vit, img_size=img_size)
    best_model = torch.load(get_model(model_dir))['state_dict']
    
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    
    test_dirs = glob.glob(test_dir + '*.pkl')
    test_set = datasets.OrcaScoreDataset(test_dirs, transforms=test_composed, output_size=img_size)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    ids = [i.split('/')[-1].split('.pkl')[0] for i in test_dirs]
    
    if cascades == 1:
        align(model, ids, test_loader, model_dir)
    elif cascades > 1:
        for i in range(cascades):
            print(f'{'-'*5} Cascade {i + 1} {'-'*5}')
            align(model, ids, test_loader, model_dir, i + 1)
            test_dir = model_dir
            test_dirs = vit_pickles(test_dir, i + 1)
            ids = [i.split('/')[-3] for i in test_dirs]
            test_set = datasets.OrcaScoreDataset(test_dirs, transforms=test_composed, output_size=img_size)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        print('Invalid number of cascades')
        return


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