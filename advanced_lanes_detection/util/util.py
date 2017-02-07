import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image_float(file_path):
    img = plt.imread(file_path)
    if type(img[0][0][0]) == np.uint8:
        img = img.astype(np.float32) / 255.0

    return img