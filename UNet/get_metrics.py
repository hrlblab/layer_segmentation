import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from pathlib import Path
import os
from tqdm import tqdm


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def per_class_dice(hist):
    return 2 * np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, nam=None):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    per_iou_list = []
    per_dice_list = []
    per_pa_list = []
    label_name_list = []
    columns = ["barckground", "inner medulla", 'inner stripe', 'outer stripe', 'cortex', 'human_cortex', 'human_medulla']

    for ind in tqdm(range(len(gt_imgs))):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]).convert('L'))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        hist_1 = fast_hist(label.flatten(), pred.flatten(), num_classes)
        iou_1 = per_class_iu(hist_1)
        dice_1 = per_class_dice(hist_1)
        pa_1 = per_Accuracy(hist_1)

        per_iou_list.append(iou_1)
        per_dice_list.append(dice_1)
        per_pa_list.append([pa_1])  # PA 应该是一个标量值，因此将其作为单独的列表元素添加
        label_name_list.append(Path(pred_imgs[ind]).name)

        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )

    final_iou_data = pd.DataFrame(per_iou_list, index=label_name_list, columns=columns)
    final_dice_data = pd.DataFrame(per_dice_list, index=label_name_list, columns=columns)
    # final_pa_data = pd.DataFrame(per_pa_list, index=label_name_list, columns=["Pixel Accuracy"])

    final_iou_data.to_csv('./result/separate/multi-test_data-iou_{}.csv'.format(nam))
    final_dice_data.to_csv('./result/separate/multi-test_data-dice_{}.csv'.format(nam))

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    Dice = per_class_dice(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                  + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                round(Precision[ind_class] * 100, 2)) + '; Dice-' + str(round(Dice[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int32), IoUs, PA_Recall, Precision, Dice

VOCdevkit_path = './layer_seg/voc_combined_dataset'

gt_dir = os.path.join(VOCdevkit_path, "SegmentationClass/")
pred_dir = './detection-results'
num_classes = 7

with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test_medulla.txt"), "r") as f:
    val_lines_mo1 = [line.strip() for line in f.readlines()]

png_name_list = val_lines_mo1
hist, IoUs, PA_Recall, Precision, Dice = compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, None, 'medulla')
