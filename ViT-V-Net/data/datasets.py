import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload, stik_load
import matplotlib.pyplot as plt
from utils import zoom_img, reduce_image

import numpy as np


class OrcaScoreData(Dataset):
    def __init__(self, data_path, y_shape = (64, 512, 512), output_size = (64, 512, 512), image_type = 'normal', reshape_mode='zoom', extension='nib'):
        self.paths = data_path
        self.y_shape = y_shape
        self.output_size = output_size
        self.reshape_mode = reshape_mode
        self.image_type = image_type
        self.extension = extension

    def __getitem__(self, index):
        path = self.paths[index]
        id = path.split('/')[-1].split('.')[0]
        
        
        if self.extension == 'nib':
            x = stik_load(path)

            x = np.rot90(x, 2, axes=(1, 2))
            x = np.flip(x, axis=2)
        elif self.extension == 'pkl':
            x = pkload(path)
        
        
        
        match self.image_type:
            case 'normal':
                if self.reshape_mode == 'zoom':
                    x = zoom_img(x, self.output_size)
                elif self.reshape_mode == 'reduce':
                    x = reduce_image(x, self.output_size)
            case 'transformed':
                if self.reshape_mode == 'zoom':
                    x = zoom_img(x, self.y_shape[index])
                elif self.reshape_mode == 'reduce':
                    x = reduce_image(x, self.y_shape[index])
                x = zoom_img(x, self.output_size)
            case _:
                x = zoom_img(x, self.output_size)
        x = x[None, ...]
        x = x.astype(np.float32)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        return {'x': x, 'id': id}

    def __len__(self):
        return len(self.paths)
    

class OrcaScoreDataset(Dataset):
    def __init__(self, data_path, transforms, output_size):
        self.paths = data_path
        self.transforms = transforms
        self.output_size = output_size

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        x = zoom_img(x, self.output_size)
        y = zoom_img(y, self.output_size)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)