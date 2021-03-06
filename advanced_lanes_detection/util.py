import matplotlib.pyplot as plt
import numpy as np


def load_image_float(file_path):
    img = plt.imread(file_path)
    return img_to_float(img)


def img_to_float(img):
    if img.dtype == np.float32:
        return img

    return img.astype(np.float32) / 255.0


def img_to_int(img):
    if img.dtype == np.uint8:
        return img

    return (img * 255).astype(np.uint8)
