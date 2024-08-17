import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory, EvalCallback_test
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    num_classes = 5
    backbone = "mobilenet"
    pretrained = False
    downsample_factor = 8

    model_path = "./logs/mouse/ep040-loss0.421-val_loss0.423.pth"

    input_shape = [1024, 1024]

    VOCdevkit_path = './layer_seg/voc_mouse_data'

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    val_lines = {
        'mouse_separate': val_lines
    }

    model = model.eval()

    log_dir = './logs/mouse/predicts'
    eval_callback = EvalCallback_test(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, eval_flag=1, period=1)
    eval_callback.on_epoch_end(1, model)