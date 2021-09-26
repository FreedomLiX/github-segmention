# -*- coding: utf-8 -*-
# @Time: 2021/6/3

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
from PIL import Image


class MeterDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 3
        self.palette = palette.Meter_palette  # 选择类别着色样式。在utils/palette.py中定义
        super(MeterDataset, self).__init__(**kwargs)

    def _set_files(self):
        file_list = os.path.join(self.root, self.split + ".txt")  # self.split是train或val
        self.files = [line.strip() for line in tuple(open(file_list, "r"))]
        self.images = [self.root+'/'+line.split(' ')[0] for line in self.files]
        self.labels = [self.root+'/'+line.split(' ')[1] for line in self.files]

    def _load_data(self, index):
        image_id = self.images[index].split('/')[-1].split('.')[0]  # 图像名称

        image_path = self.images[index]  # 图像文件路径
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)  # 图像文件

        mask_path = self.labels[index]  # 标签文件路径
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8)  # 标签文件

        return image, mask, image_id


class Meter(BaseDataLoader):
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

        self.dataset = MeterDataset(**kwargs)
        super(Meter, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
