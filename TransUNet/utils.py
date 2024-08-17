import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from skimage import io


# class DiceLoss(nn.Module):
#     def __init__(self, n_classes=2):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
#
#     def forward(self, inputs, target, weight=None, softmax=True):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target)
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes


def DiceLoss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
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


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    tmp = np.zeros_like(pred, dtype=np.uint8)
    tmp[0] = gt
    tmp[1] = gt
    tmp[2] = gt
    gt = tmp
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    if len(image.shape) == 3:
        # prediction = np.zeros_like(label)
        prediction = np.zeros([3, *label.shape], dtype=np.uint8)
        # for ind in range(image.shape[0]):
        #     slice = image[ind, :, :]
        #     x, y = slice.shape[0], slice.shape[1]
        #     # x, y = slice.shape[1], slice.shape[2]
        #     if x != patch_size[0] or y != patch_size[1]:
        #         slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        #     input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        #     print(input.shape)
        net.eval()
        with torch.no_grad():
            outputs = net(image)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction = pred
        io.imsave(f'{test_save_path}/{case}.png', prediction)
    else:
        input = image.unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy().astype(np.int32)
            io.imsave(f'{test_save_path}/{case}.png', prediction)
    # metric_list = []
    # for i in range(1, classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))
    #
    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     # sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     # sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     # sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    # return metric_list