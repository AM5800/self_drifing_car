import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import util


def apply_to_test_images(fun):
    out_dir = "out"

    for img_path in glob.glob("input/test_images/*"):
        img = util.load_image_float(img_path)
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


def threshold_lines(img, mag_threshold=(0.1, 0.8), dir_threshold=(0.7, 1.3)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = hsv[:, :, 2]

    alpha = 100
    beta = -70

    contrast = np.clip(alpha * gray + beta, 0.0, 1.0)

    ksize = 15
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_factor = np.max(magnitude)

    magnitude = threshold(magnitude / scale_factor, mag_threshold[0], mag_threshold[1])

    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    direction = threshold(direction, dir_threshold[0], dir_threshold[1])

    return direction * magnitude * contrast


if __name__ == "__main__":
    apply_to_test_images(threshold_lines)
