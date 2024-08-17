import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.pspnet import PSPNet
from nets.pspnet_training import (get_lr_scheduler, set_optimizer_lr,
                                  weights_init)
from utils.callbacks import LossHistory, EvalCallback_test
from utils.dataloader import PSPnetDataset, pspnet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    num_classes = 2
    backbone = "resnet50"
    pretrained = False

    model_path = "./logs/cortex_separate/ep005-loss0.279-val_loss0.340.pth"

    input_shape = [1024, 1024]

    VOCdevkit_path = './layer_seg/voc_human_data/cortex'
    downsample_factor = 8
    aux_branch = False

    model = PSPNet(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                   pretrained=pretrained, aux_branch=aux_branch)

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    model = model.eval()

    val_lines = {
        'cortex_separate': val_lines
    }

    log_dir = './logs/mouse/predicts'
    eval_callback = EvalCallback_test(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, eval_flag=1, period=1)
    eval_callback.on_epoch_end(1, model)
