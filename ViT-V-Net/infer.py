import glob
from torch.utils.tensorboard import SummaryWriter
import logging
import os, losses, utils, nrrd
import shutil
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from torchsummary import summary
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
import argparse
import pandas as pd



def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], linewidth=0.8, **kwargs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))

def SSMI(x, y):
    max_x = x.max()
    min_x = x.min()
    max_y = y.max()
    min_y = y.min()
    return ssim(x, y, data_range=max(max_x, max_y) - min(min_x, min_y))

def update_results(results, moved, fixed, deformed, index, direction):
    results['mae']['pre'].append(MAE_torch(moved, fixed).item())
    results['mae']['post'].append(MAE_torch(deformed, fixed).item())
    
    results['rmse']['pre'].append(torch.sqrt(MSE_torch(moved, fixed)).item())
    results['rmse']['post'].append(torch.sqrt(MSE_torch(deformed, fixed)).item())
    
    results['ssim']['pre'].append(SSMI(moved, fixed))
    results['ssim']['post'].append(SSMI(deformed, fixed))
    
    results['index'].append(index)
    results['direction'].append(direction)
    
    return results

def args_input():
    parser = argparse.ArgumentParser(description='ViT-V-Net')
    parser.add_argument('--test_dir', type=str, default='/vit-v-net/test/', help='testing data directory')
    parser.add_argument('--model_folder', type=str, default='vit-v-net/', help='model directory')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    
    return parser.parse_args()


def show_metrics(results, index):
    print(f'Image: {index} | Direction: {results["direction"][-1]}')
    print(f' - MAE (pre) {results["mae"]["pre"][-1]:.4f} | RMSE (pre) {results["rmse"]["pre"][-1]:.4f} | SSIM (pre) {results["ssim"]["pre"][-1]:.4f}')
    print(f' - MAE (post) {results['mae']['post'][-1]:.4f} | RMSE (post) {results['rmse']['post'][-1]:.4f} | SSIM (post) {results['ssim']['post'][-1]:.4f}')
    
def show_results(df):
    print(f"MAE before {np.mean(df['mae_pre'])}, after {np.mean(df['mae_post'])}")
    print(f"RMSE before {np.mean(df['rmse_pre'])}, after {np.mean(df['rmse_post'])}")
    print(f"SSIM before {np.mean(df['ssim_pre'])}, after {np.mean(df['ssim_post'])}")

def main():
    args = args_input()
    test_dir = args.test_dir
    model_idx = -1
    model_folder = args.model_folder
    model_dir = 'experiments/' + model_folder
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    batch_size = args.batch_size
    num_workers = args.num_workers
    img_size = (64, 512, 512)
    # dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        os.remove('experiments/'+model_folder[:-1]+'.csv')
    # csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    # line = ''
    # for i in range(46):
    #     line = line + ',' + dict[i]
    # csv_writter(line, 'experiments/' + model_folder[:-1])
    model = models.ViTVNet(config_vit, img_size=img_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.OrcaScoreDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed, output_size=img_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    test_results = {
        'index': [],
        'direction': [],
        'mae': {'pre': [], 'post': []},
        'rmse': {'pre': [], 'post': []},
        'ssim': {'pre': [], 'post': []}
    }
    
    with torch.no_grad():
        # stdy_idx = 0
        for i, data in enumerate(test_loader):
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_seg = data[2]
            # y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            
            test_results = update_results(test_results, x, y, x_def, i, 'XtoY')
            show_metrics(test_results, i)
            
            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            #jac_det = utils.jacobian_determinant(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            # line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            #eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans diff: {:.4f}, Raw diff: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            y_def, flow = model(y_in)
            test_results = update_results(test_results, y, x, y_def, i, 'YtoX')
            show_metrics(test_results, i)
            # def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            # tar = x.detach().cpu().numpy()[0, 0, :, :, :]

            #jac_det = utils.jacobian_determinant(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            # line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            # out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            #print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            #eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            # dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans diff: {:.4f}, Raw diff: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

        df_data = {
            'image_index': test_results['index'],
            'direction': test_results['direction'],
            'mae_pre': test_results['mae']['pre'],
            'mae_post': test_results['mae']['post'],
            'rmse_pre': test_results['rmse']['pre'],
            'rmse_post': test_results['rmse']['post'],
            'ssim_pre': test_results['ssim']['pre'],
            'ssim_post': test_results['ssim']['post']
        }
        
        df = pd.DataFrame(df_data)
        
        print(f'\n\n{'-'*10} X to Y summary {'-'*10}')
        show_results(df[df['direction'] == 'XtoY'])
        print(f'\n\n{'-'*10} Y to X summary {'-'*10}')
        show_results(df[df['direction'] == 'YtoX'])
        
        
        df.to_csv('experiments/' + model_folder[:-1] + '.csv', index=False)
        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

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