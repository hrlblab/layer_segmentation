import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
import torch.nn as nn

import os
import datetime
import torch.distributed as dist
import torch.optim as optim
from utils.callbacks import LossHistory, EvalCallback_test


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./layer_seg/transunet_combined_data/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='cortex', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', default='./model_out', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=61, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=10, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--cfg', default='configs/swin_tiny_patch4_window7_224_lite.yaml', type=str, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)


if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    backbone = "resnet50"
    pretrained = False
    num_classes = 7

    model_path = "./layer_seg/swin-unet/model_out/combined/epoch_26.pth"

    dataset_name = args.dataset
    dataset_config = {
        'cortex': {
            'root_path': './layer_seg/transunet_combined_data/train_npz',
            'list_dir': './layer_seg/transunet_combined_data/lists',
            'num_classes': 7,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    # model.load_from(config)
    model = nn.DataParallel(model)

    input_shape = [1024, 1024]

    VOCdevkit_path = './layer_seg/voc_combined_dataset'

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    model = model.eval()

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test_medulla.txt"), "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    val_lines = {
        'cortex': val_lines
    }

    log_dir = './model_out'
    eval_callback = EvalCallback_test(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, eval_flag=1, period=1)
    eval_callback.on_epoch_end(1, model)