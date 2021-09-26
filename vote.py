# -*- coding: utf-8 -*-
# @Time: 2021/5/26

import argparse
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2] // 2.5), int(image_size[3] // 2.5))
    overlap = 1 / 3

    stride = ceil(tile_size[0] * (1 - overlap))

    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, mask, output_path, image_file, palette):
    # Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file + '.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))


def main():
    # 接收命令行参数。model和config是要集成的模型和配置文件列表
    args = parse_arguments()

    # 图像文件列表
    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    # 检查output路径
    if not os.path.exists(args.output):
        print('-----make output dir-----')
        os.mkdir(args.output)

    pred_li=[]
    palette_final=None
    # 每个模型加载一次
    for i in range(len(args.config)):
        # 加载参数中的config.json文件
        config = json.load(open(args.config[i]))

        # Dataset used for training the model  模型的训练集
        dataset_type = config['train_loader']['type']
        assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'TianChi']  # insert item 'TianChi'
        if dataset_type == 'CityScapes':
            scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        else:
            scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        # 训练集的相关配置
        loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(loader.MEAN, loader.STD)
        num_classes = loader.dataset.num_classes
        palette = loader.dataset.palette
        palette_final = palette  # 记录着色方式。TODO: 每加载一个模型都会发生更新，但由于模型训练集相同(针对同一任务)，着色方式也相同

        # Model  模型的网络结构
        model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
        # 设置gpu
        availble_gpus = list(range(torch.cuda.device_count()))
        device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        # Load checkpoint  加载模型
        checkpoint = torch.load(args.model[i], map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        # If during training, we used data parallel  训练中使用了数据并行
        if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
            # for gpu inference, use data parallel
            if "cuda" in device.type:
                model = torch.nn.DataParallel(model)
            else:
                # for cpu inference, remove module
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint = new_state_dict

        # load  载入
        model.load_state_dict(
            checkpoint,
            False
        )  # insert parameter 'False' in order to ignore the error RuntimeError: Error(s) in loading state_dict for DataParallel: Missing key(s) in state_dict: "module.initial.0.0.weight", "module.initial.0.1.weight", ...

        model.to(device)

        # evaluation
        model.eval()

        pred_imgs=[]
        with torch.no_grad():
            tbar = tqdm(image_files, ncols=100)  # 在进度条中依次读取图像文件并推理
            for img_file in tbar:
                image = Image.open(img_file).convert('RGB')
                input = normalize(to_tensor(image)).unsqueeze(0)

                if args.mode == 'multiscale':
                    prediction = multi_scale_predict(model, input, scales, num_classes, device)
                elif args.mode == 'sliding':
                    prediction = sliding_predict(model, input, num_classes)
                else:
                    prediction = model(input.to(device))
                    prediction = prediction.squeeze(0).cpu().numpy()
                # print(prediction)
                # type=<class 'numpy.ndarray'> shape=(2,512,512)
                prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
                # type=<class 'numpy.ndarray'> shape=(512,512)

                pred_imgs.append(prediction)
        pred_li.append(pred_imgs)

    for i in range(len(image_files)):
        image = Image.open(image_files[i]).convert('RGB')  # 依次读取图像文件 对推理结果求平均 并储存

        pred_img=np.zeros(pred_li[0][i].shape)
        for j in range(len(pred_li)):  # 计算每个模型j对于一张图像i的推理结果
            pred_img+=pred_li[j][i]

        # 多个模型预测结果集成
        pred_img = pred_img / float(len(pred_li))  # TODO: 求平均或设置不同权重

        # 保存推理后图像
        save_images(image, pred_img, args.output, image_files[i], palette_final)
        print('saved one image')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config',
                        default=['./saved/UNet_20210520/05-20_16-59/config.json',
                                 './saved/UNet_20210520/05-20_16-59/config.json',
                                 './saved/UNet_20210520/05-20_16-59/config.json'],
                        type=list,
                        help='The config used to train the model')  # 模型训练的配置文件
    parser.add_argument('-mo', '--mode',
                        default='multiscale',
                        type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')  # 用于推理的模式
    parser.add_argument('-m', '--model',
                        default=['./saved/UNet_20210520/05-20_16-59/best_model.pth',
                                 './saved/UNet_20210520/05-20_16-59/checkpoint-epoch150.pth',
                                 './saved/UNet_20210520/05-20_16-59/checkpoint-epoch140.pth'],
                        type=list,
                        help='Path to the .pth model checkpoint to be used in the prediction')  # 模型
    parser.add_argument('-i', '--images',
                        default=None,
                        type=str,
                        help='Path to the images to be segmented')  # 待推理图片路径
    parser.add_argument('-o', '--output',
                        default='outputs',
                        type=str,
                        help='Output Path')  # 推理结果图片输出路径
    parser.add_argument('-e', '--extension',
                        default='jpg',
                        type=str,
                        help='The extension of the images to be segmented')  # 待推理图像类型
    args = parser.parse_args()
    return args


if __name__=='__main__':
    main()
