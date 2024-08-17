import glob
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import imageio

import os
import glob
import numpy as np
import imageio.v3
from tqdm import tqdm
from pathlib import Path


def npz():
    # 图像路径
    path = './layer_seg/voc_combined_dataset/JPEGImages/*.jpg'
    out_root = './layer_seg/transunet_combined_data/train_npz'
    os.makedirs(out_root, exist_ok=True)

    for i, img_path in enumerate(tqdm(glob.glob(path))):
        # 读入图像
        image = imageio.v3.imread(img_path)

        # 读入标签
        label_path = img_path.replace('JPEGImages', 'SegmentationClass')
        label_path = label_path.replace(label_path[-3:], 'png')

        if not os.path.exists(label_path):
            # 路径不存在，跳过
            print(f"Path does not exist: {label_path}")
            continue
        else:
            label = imageio.v3.imread(label_path, mode='L')

            # 如果标签中有值为11的像素，将其替换为6
            label[label == 11] = 6

            # 保存npz
            out_path = os.path.join(out_root, Path(img_path).name[:-4])
            np.savez(out_path, image=image, label=label)


if __name__ == '__main__':
    npz()


