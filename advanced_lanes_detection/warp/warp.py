import cv2
import matplotlib.pyplot as plt
import numpy as np


def warp(img):
    top = 500
    bottom = 700
    xcenter = img.shape[1] / 2
    top_half_width = 170
    bottom_half_width = 600

    src = np.float32(
        [[xcenter - top_half_width, top],
         [xcenter + top_half_width, top],
         [xcenter - bottom_half_width, bottom],
         [xcenter + bottom_half_width, bottom]])

    dst = np.float32(
        [[0, top],
         [img.shape[1], top],
         [0, bottom],
         [img.shape[1], bottom]])

    M = cv2.getPerspectiveTransform(src, dst)

    dsize = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, dsize)
    return warped


if __name__ == "__main__":
    result = warp(plt.imread("../test_images/vlcsnap-2017-02-02-08h52m36s50.png"))
    plt.imshow(result)
    plt.show()
