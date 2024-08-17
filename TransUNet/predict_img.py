import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from utils.callbacks import LossHistory, EvalCallback_test

import argparse
import logging
import os
import random
import torch
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./layer_seg/transunet_mouse_data/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='mouse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.002,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    backbone = "resnet50"
    pretrained = False
    num_classes = 2

    model_path = "./layer_seg/transunet/model/TU_cortex1024/TU_pretrain_R50-ViT-B_16_skip3_30k_epo75_bs4_lr0.002_1024/epoch_3.pth"

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    model = ViT_seg(config_vit, img_size=1024, num_classes=2)

    model = nn.DataParallel(model)

    input_shape = [1024, 1024]

    VOCdevkit_path = './layer_seg/voc_human_data/cortex'

    model.load_state_dict(torch.load(model_path))

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    val_lines = {
        'cortex': val_lines
    }

    log_dir = './model/TU_medulla1024/medulla_predict'
    eval_callback = EvalCallback_test(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, eval_flag=1, period=1)
    eval_callback.on_epoch_end(1, model)



