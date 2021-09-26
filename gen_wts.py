import torch
from collections import OrderedDict
import struct

# Load checkpoint
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('/home/dixn/fsdownload/best_model.pth', map_location=device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']

# 去除 moudle.
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]
    new_state_dict[name] = v
# 生成.wts 文件
f = open('avgPool.wts', 'w')
f.write('{}\n'.format(len(new_state_dict.keys())))
for k, v in new_state_dict.items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    print(k, v.shape)
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f', float(vv)).hex())
    f.write('\n')
