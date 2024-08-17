import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
# from utils import DiceLoss
from torchvision import transforms
from utils.callbacks import EvalCallback

def DiceLoss(inputs, target, beta=1, smooth=1e-5):
    target = target.unsqueeze(1).repeat(1, 2, 1, 1)
    n, c, h, w = inputs.size()
    nt, ct, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def iou(pred, target, n_classes=2, softmax=True):
    if softmax:
        pred = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.nanmean(ious)


def trainer_synapse(args, model, snapshot_path):
    # device_ids = [1]
    # device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    db_test = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test_vol")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.00001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    total_iou = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    VOCdevkit_path = './layer_seg/voc_human_data/cortex'
    eval_flag = True
    eval_period = 5

    input_shape = [1024, 1024]

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines_mo1 = [line.strip() for line in f.readlines()]
    # with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test_medulla.txt"), "r") as f:
    #     val_lines_me = [line.strip() for line in f.readlines()]
    # with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test_cortex.txt"), "r") as f:
    #     val_lines_co = [line.strip() for line in f.readlines()]

    # val_lines = {
    #     'mouse': val_lines_mo1, 'medulla': val_lines_me, 'cortex': val_lines_co
    # }

    val_lines = {
        'cortex': val_lines_mo1
    }
    Cuda = True

    log_dir = './layer_seg/swin-unet/model_out/cortex2'

    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
                                 eval_flag=eval_flag, period=eval_period)


    # iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in tqdm(range(max_epoch)):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = DiceLoss(outputs, label_batch)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            total_iou += iou(outputs, label_batch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            miou_train = total_iou / iter_num
            writer.add_scalar('info/miou_train', miou_train, iter_num)
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            if iter_num % 10 == 0:
                logging.info('iteration %d : loss_dice : %f, loss_ce: %f' % (
                iter_num, loss_dice.item(), loss_ce.item()))
            #
            # if iter_num % 3000 == 0:
            #     tmp_iou = 0
            #     for i_test_batch, sampled_test_batch in tqdm(enumerate(testloader)):
            #         image_batch, label_batch = sampled_test_batch['image'], sampled_test_batch['label']
            #         with torch.no_grad():
            #             image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            #             outputs = model(image_batch)
            #             miou_test = iou(outputs, label_batch)
            #             tmp_iou += miou_test
            #             writer.add_scalar('info/miou_test', miou_test, iter_num)
            #
            #     logging.info('iteration %d : loss_dice : %f, loss_ce: %f, miou_test: %f' % (iter_num, loss_dice.item(), loss_ce.item(), tmp_iou/len(testloader)))


        # save_interval = 50  # int(max_epoch/6)
        eval_callback.on_epoch_end(epoch_num+1, model)
        save_freq = 1
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if epoch_num % save_freq == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break

    writer.close()
    return "Training Finished!"