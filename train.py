import os
import json
import argparse
import torch
import dataloaders
import models
import logging
from utils import losses
from utils import Logger
from trainer import Trainer

# from trainer_smp import create_dataloader_meter, model_build, train


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')
    # type=<class 'models.unet.UNet'>

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    # TRAINING
    trainer = Trainer(model=model,
                      loss=loss,
                      resume=resume,
                      config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      train_logger=train_logger
                      )
    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config',
                        default='config.json',
                        type=str,
                        help='Path to the config file (default: config.json)')  # .json格式配置文件
    parser.add_argument('-r', '--resume',
                        default=None,
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')  # 中断后恢复训练使用的ckpt文件位置
    parser.add_argument('-d', '--device',
                        default=None,
                        type=str,
                        help='indices of GPUs to enable (default: all)')  # 使用的gpu
    args = parser.parse_args()

    config = json.load(open(args.config))  # 读取配置文件
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
