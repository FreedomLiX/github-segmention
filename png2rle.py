# -*- coding: utf-8 -*-
# @Time: 2021/5/24

'''将png推理结果图像转换为rle编码格式，适用于aliyun竞赛test集的最后提交格式。
   注意最终提交csv文件中图像的顺序需要以官方样例test_a_samplesubmit.csv为准。'''

from dataloaders.tianchi import rle_encode, rle_decode
from PIL import Image
import numpy as np
import pandas as pd
import tqdm


test_mask = pd.read_csv('../aliyun/TianChi/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: '../aliyun/test_a_output/' + x)

li=[]
pbar = tqdm.tqdm()
for idx, name in enumerate(test_mask['name'].iloc[:]):
    img_name = name[:-3]+'png'  # 实际推理结果图像的路径
    img = np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32)  # 读取图像并转为rle编码格式
    # print(img.shape)  # (512, 512, 3)。(0,0,0)黑，(255,255,255)白

    img_2 = img[:, :, 0]
    # for i in np.nditer(img_2, op_flags=['writeonly']):
    #     if i==255: i=1
    img_rle = rle_encode(img_2)  # rle编码结果
    # 根据rle_encode()函数实现，不需要转为0-1形式再进行编码，直接调用编码即可
    # 语义分割结果图像若全黑（无建筑物，全为background），则rle编码结果为''

    li.append([name.split('/')[-1], img_rle])
    pbar.update()

li = pd.DataFrame(li)
li.to_csv('../aliyun/test_a_result_8.csv', index=None, header=None, sep='\t')

print('saved in csv file.')


'''
# 测试编解码是否正确
<<<<<<< HEAD
test_mask = pd.read_csv('../aliyun/test_a_result_7.csv', sep='\t', names=['name', 'mask'])
=======
test_mask = pd.read_csv('../aliyun/test_a_result_2.csv', sep='\t', names=['name', 'mask'])
>>>>>>> 39d8d67b952b4a635e73330e81efe718434cada5

# 读取第一张图，并将对应的rle解码为mask矩阵
img = np.asarray(Image.open(output_path + test_mask['name'].iloc[0][:-3] + 'png').convert('RGB'), dtype=np.float32)  # 图像文件
print(img.shape)  # (512, 512, 3)
print(type(img))  # <class 'numpy.ndarray'>

mask = rle_decode(test_mask['mask'].iloc[0])

print(rle_encode(mask) == test_mask['mask'].iloc[0])  # True
print(rle_encode(mask) == rle_encode(img[:,:,0]))  # True
'''
