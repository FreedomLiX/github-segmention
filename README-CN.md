# pytorch-segmentation

原工程来自: https://github.com/yassouali/pytorch-segmentation 
包含对多个语义分割网络的pytorch实现。



## 下载和安装

根据requirements.txt中要求安装所需包。torch，torchvision，PIL，opencv，tqdm，tensorboard等。

```bash
$ cd pytorch-segmentation/
$ pip install -r requirements.txt
```

>所需要的ResNet预训练网络、训练好的PSPnet模型、VOC数据集、aliyun的TianChi数据集可使用百度网盘下载。
>
>链接: https://pan.baidu.com/s/1FPFD2Ys8w_85yCJShPQ51g  提取码: 770n 



## 训练

修改config.json配置文件中的参数。

```bash
$ python train.py --config config.json
```

使用tensorboard监视训练过程。

```bash
$ tensorboard --logdir saved
```



## 推理

使用一个训练好的pytorch模型，对目标文件夹下所有图像进行推理，结果文件存入outputs/。

```bash
$ python inference.py --config config.json --model PSPnet.pth --images tests/
```

可选参数如下。

```
--output       推理结果图像存储路径。默认outputs。
--extension    待推理图像扩展名。默认jpg。
--images       待推理图像存储路径。
--model        训练好的pytorch模型。
--mode         推理模式选择：multiscale 或 sliding。默认multiscale。
--config       训练该模型时的配置文件。
```



## \*模型融合

仅针对aliyun竞赛，使用TianChi数据集。将不同模型结构的推理后结果进行集成，得到融合结果的语义分割图像。

```bash
$ python vote.py
```

可选参数与inference.py相同，需要注意，其中--config和--model参数是列表形式，表示需要集成的模型使用的配置文件和模型文件。



## \*转rle编码

仅针对aliyun竞赛，使用TianChi数据集。根据推理后图像，转为rle编码格式，生成test_a_result.csv文件并提交。

```bash
$ python png2rle.py 
```



## \*调用smp库用于模型训练训练

由于原工程支持的语义分割网络结构和主干网数量有限，这里可调用smp包中封装好的模型结构进行训练。

安装命令如下。

```bash
$ pip install segmentation-models-pytorch
$ pip install albumentations
```

需将config.json中的参数项"smp"置为true，并设置"smp_args"中的参数项，再运行train.py即可。

>注意，smp库是为扩展模型训练所使用网络结构所添加，其具体训练程序与原工程不同，因此除"smp_args"所包含以外的配置参数在smp为true时不生效。



## \*对使用smp库构建的网络训练出的模型进行推理

在执行完smp参数置为True的train.py后，得到训练好的pytorch模型，运行推理代码直接生成语义分割结果的rle编码文件。

```bash
$ python inference_smp.py
```



## config.json配置文件参数

```
{
    "smp": true,  // 是否调用smp库进行模型训练
    "smp_args": {  // 使用smp库进行模型训练时的相应参数
        "type": "DeepLabV3Plus",  //支持Unet, UnetPlusPlus, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus
        "backbone": "xception",  //支持ResNet, ResNeXt, ResNeSt, Res2Ne(X)t, RegNet(x/y), SE-Net, SK-ResNe(X)t, DenseNet, Inception, EfficientNet, MobileNet, DPN, VGG网络系列
        "weights": "imagenet",  // 预训练主干网初始化使用的权重类型, None或imagenet或其他支持的权重
        "data_dir": "../meter_seg",
        "classes": 2,
        "epochs": 50,
        "batch_size": 2,
        "image_size": 512,
        "device": "cuda"
    },

    "name": "PSPNet",  // 训练会话名称
    "n_gpu": 1,  // 用于训练的gpu数量
    "use_synch_bn": false,  // 使用同步的batchnorm（用于多gpu时）

    "arch": {
        "type": "PSPNet",  // 使用的网络结构名称
                           // 支持FCN8, UNet, UNetResnet, SegNet, SegResNet, ENet, GCN, DeepLab, DeepLab_DUC_HDC, UperNet, PSPNet, PSPDenseNet
        "args": {
            "backbone": "resnet50",  // 使用的encoder类型（主干网）
            "freeze_bn": false,  // 进行fine-tuning微调时使用
            "freeze_backbone": false  // 仅训练decoder时使用
        }
    },

    "train_loader": {
        "type": "TianChi",  // 使用的dataloader类型（数据集）
        "args":{
            "data_dir": "../aliyun",  // 训练集路径
            "batch_size": 4,  // 批处理大小
            "base_size": 256,  // 将图像resize的大小
            "crop_size": 256,  // 重新缩放后随机裁剪的大小
            "augment": true,  // 数据增强
            "shuffle": true, 
            "scale": true,
            "flip": true,
            "rotate": true,
            "blur": false,
            "split": "train",  // 拆分训练集验证集...
            "num_workers": 8
        }
    },

    "val_loader": {  // 与训练集相同，但不进行数据扩充，仅中心裁剪
        "type": "TianChi",
        "args":{
            "data_dir": "../aliyun",
            "batch_size": 8,
            "crop_size": 480,
            "val": true,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,  // Using lr/10 for the backbone, and lr for the rest
        "args":{
            "lr": 0.01,  // 学习率
            "weight_decay": 1e-4,  // 衰减率
            "momentum": 0.9
        }
    },

    "loss": "CrossEntropyLoss2d",  // 损失函数（utils/losses.py）
    "ignore_index": 255,  // Class to ignore (must be set to -1 for ADE20K) dataset
    "lr_scheduler": {  // 学习率调整策略
        "type": "Poly",
        "args": {}
    },

    "trainer": {
        "epochs": 150,  // 训练轮数
        "save_dir": "saved/",  // ckpt模型文件保存路径
        "save_period": 10,  // 每经过多少轮保存一次
  
        "monitor": "max Mean_IoU",  // 模型性能的模式和指标
        "early_stop": 10,  // Number of epochs to wait before early stoping (0 to disable)
        
        "tensorboard": true,  // 可视化
        "log_dir": "saved/runs",  // 日志存储路径
        "log_per_iter": 1,  // 经过多少轮迭代（iteration）记录一次日志

        "val": false,  // 是否进行验证
        "val_per_epochs": 5  // 每经过多少轮验证一次
    }
}
```



**数据集**：

* TianChi：aliyun竞赛数据集。仅提供训练集train/和测试集test_a/，因此训练中可将验证过程参数val置为False。
* Meter：百度paddle提供的仪表数据集，包含train，val，test部分，位于meter_seg/目录。
* PASCAL VOC：下载VOC 2012并解压得到VOCtrainval_11-May-2012/目录。
* ADE20K
* Cityscapes
* COCO



## 目录结构

```
pytorch-segmentation/
│
├── base/ - 基类
│   ├── base_data_loader.py
│   ├── base_dataset.py - 原工程数据增强在此实现
│   ├── base_model.py
│   └── base_trainer.py
│
├── dataloader/ - 从不同的语义数据集中加载数据
│   ├── labels/ - 不同数据集的类别txt
│   │   └── ...
│   ├── ade20k.py
│   ├── cityscapes.py
│   ├── coco.py
│   ├── meter.py
│   ├── tianchi.py
│   └── voc.py
│
├── models/ - 原工程使用的各类语义分割网络结构
│   └── ...
│
├── pretrained/ - 预训练模型
│   ├── resnet18-5c106cde.pth
│   ├── resnet34-333f7ec4.pth
│   ├── resnet50s-a75c83cf.pth
│   ├── resnet101s-03a0f310.pth
│   └── resnet152s-36670e8b.pth
│
├── saved/ - 训练日志和保存下来的模型
│   └── ...
│  
├── utils/ - 其他函数
│   └── ...
│
├── config.json - 配置文件
│
├── inference.py - 推理
├── inference_smp.py - 对于smp库训练成的模型进行推理
│
├── png2rle.py - 将语义分割结果图像转为rle编码格式
│
├── train.py - 训练
├── trainer.py - 训练具体实现
├── trainer_smp.py - 调用smp库进行训练时的具体实现
│
├── vote.py - 模型融合（投票法）
│
└── ...
```





