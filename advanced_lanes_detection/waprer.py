import cv2
import matplotlib.pyplot as plt
import numpy as np


class Warper:
    def __init__(self, img_shape):
        self.__img_shape = img_shape
        top = 500
        bottom = 700
        xcenter = img_shape[1] / 2
        top_half_width = 170
        bottom_half_width = 600

        self.__src = np.float32(
            [[xcenter - top_half_width, top],
             [xcenter + top_half_width, top],
             [xcenter - bottom_half_width, bottom],
             [xcenter + bottom_half_width, bottom]])

        self.__dst = np.float32(
            [[0, top],
             [img_shape[1], top],
             [0, bottom],
             [img_shape[1], bottom]])

    def warp(self, img):
        return self.__warp(img, self.__src, self.__dst)

    def __warp(self, img, src, dst):
        if img.shape != self.__img_shape:
            raise Exception("Invalid image size. Expected {0}, but got {1}" % img.shape, self.__img_shape)

        M = cv2.getPerspectiveTransform(src, dst)

        dsize = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, dsize)
        return warped

    def unwarp(self, img):
        return self.__warp(img, self.__dst, self.__src)


if __name__ == "__main__":
    img = plt.imread("test_images/vlcsnap-2017-02-02-08h52m36s50.png")
    warper = Warper(img.shape)
    result = warper.warp(img)
    # result = warper.unwarp(result)
    plt.imshow(result)
    plt.show()
