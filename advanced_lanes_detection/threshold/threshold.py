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


def apply_to_test_images(fun):
    out_dir = "out"

    for img_path in glob.glob("../test_images/static/*"):
        img = load_image_float(img_path)
        result = fun(img)

        if len(result.shape) == 2 or result.shape[2] == 1:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        combined = np.concatenate([img, result])
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        plt.imsave(out_path, combined)


def threshold(data, thresh_min, thresh_max):
    result = np.zeros_like(data, np.float32)
    result[(data >= thresh_min) & (data <= thresh_max)] = 1.0
    return result


def find_lines(img, mag_threshold=(15, 100), dir_threshold=(np.pi / 7, np.pi / 2.5)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = hsv[:, :, 1]

    ksize = 3
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(magnitude)
    magnitude = (255 * magnitude / scale_factor).astype(np.uint8)
    magnitude = threshold(magnitude, mag_threshold[0], mag_threshold[1])

    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    direction = threshold(direction, dir_threshold[0], dir_threshold[1])

    return direction * magnitude


if __name__ == "__main__":
    apply_to_test_images(find_lines)
