# -*- coding: utf-8 -*-
# @Time: 2021/5/18

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


class TianChiDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.TianChi_palette  # 选择类别着色样式。在utils/palette.py中定义
        super(TianChiDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, 'train')  # 训练集图像文件夹
        self.label_file = os.path.join(self.root, 'train_mask.csv')  # 训练集标签文件
        self.train_mask = pd.read_csv(self.label_file, sep='\t', names=['name', 'mask'])
        self.files = [i for i in self.train_mask['name'].iloc]  # 图像文件名称，共30000
        self.labels = [i for i in self.train_mask['mask'].iloc]  # 图像标签的rle编码，共30000

    def _load_data(self, index):
        image_id = self.files[index].split('.')[0]  # 图像名称
        image_path = os.path.join(self.image_dir, self.files[index])  # 图像文件路径

        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)  # 图像文件

        try:
            mask = rle_decode(self.labels[index])  # 根据rle编码进行解码后的图像标签
        except:
            # 当前图像上没有建筑，无标签。则手动创建其标签。
            # 根据rle_decode()函数的返回值创建一个全0的512×512多维数组
            # print('No objects on this image, create a mask manually')
            mask = np.zeros(512*512, dtype=np.uint8).reshape((512,512), order='F')

        label = np.asarray(mask, dtype=np.int32)
        # 转换为numpy.ndarray格式的标签文件。是二维的。
        # 此处等价于，将mask转为png文件后再用plt.image.open()读取得到的矩阵。该函数读取图像后转np.array也是二维的。

        return image, label, image_id


class TianChi(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None,
                 scale=True, num_workers=1, val=False, shuffle=False, flip=False,
                 rotate=False, blur=False, augment=False, val_split=None, return_id=False):
        self.MEAN = [0.625, 0.448, 0.688]
        self.STD = [0.131, 0.177, 0.101]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = TianChiDataset(**kwargs)
        super(TianChi, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
