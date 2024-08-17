import numpy as np
from PIL import ImageEnhance, Image
import random

def random_color(image, saturation=0, brightness=0, contrast=0, sharpness=0):
    image = Image.fromarray(image)
    if random.random() < saturation:
        # print('saturation')
        random_factor = np.random.randint(0, 21) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        # print('brightness')
        random_factor = np.random.randint(10, 12) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        # print('contrast')
        random_factor = np.random.randint(10, 12) / 10.  # 随机因子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        # print('sharpness')
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return np.asarray(image, dtype=np.uint8)