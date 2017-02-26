import matplotlib.pyplot as plt
import numpy as np
from typing import List


def try_load_image_float(file_path):
    try:
        return load_image_float(file_path)
    except:
        return None


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


def parallel_shuffle(values: List[np.array]):
    if len(values) == 0:
        return []

    permutation = np.random.permutation(len(values[0]))
    result = []
    for value in values:
        result.append(value[permutation])

    return result
