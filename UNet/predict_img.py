import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback_test
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    num_classes = 5
    backbone = "resnet50"
    pretrained = False

    model_path = "./logs/mouse/ep060-loss0.283-val_loss0.243.pth"

    input_shape = [1024, 1024]

    VOCdevkit_path = './layer_seg/voc_human_data/medulla'

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    val_lines = {
        'medulla_separate': val_lines
    }

    log_dir = './logs/mouse/predicts'
    eval_callback = EvalCallback_test(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, eval_flag=1, period=1)
    eval_callback.on_epoch_end(1, model)



