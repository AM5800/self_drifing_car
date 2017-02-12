import cv2
import matplotlib.pyplot as plt
import numpy as np


class Warper:
    def __init__(self, img_shape):
        self.__img_shape = img_shape
        top = 460  # Conrols height of upper segment of trapeze
        bottom = 660  # Conrols height of bottom segment of trapeze
        xcenter = img_shape[1] / 2
        top_half_width = 107  # Half width of top part of trapeze
        bottom_half_width = 650

        self.src = np.float32(
            [[xcenter - top_half_width, top],
             [xcenter + top_half_width, top],
             [xcenter + bottom_half_width, bottom],
             [xcenter - bottom_half_width, bottom]])

        self.dst = np.float32(
            [[0, 0],
             [img_shape[1], 0],
             [img_shape[1], img_shape[0]],
             [0, img_shape[0]]])

    def warp(self, img):
        return self.__warp(img, self.src, self.dst)

    def __warp(self, img, src, dst):
        if img.shape[0:2] != self.__img_shape[0:2]:
            raise Exception("Invalid image size. Expected {0}, but got {1}".format(img.shape, self.__img_shape))

        M = cv2.getPerspectiveTransform(src, dst)

        dsize = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, dsize)
        return warped

    def unwarp(self, img):
        return self.__warp(img, self.dst, self.src)

if __name__ == "__main__":
    img = plt.imread("test_images/vlcsnap-2017-02-02-08h52m36s50.png")
    warper = Warper(img.shape)
    warped = warper.warp(img)
    cv2.polylines(img, np.int32([warper.src]), True, (0, 1, 0), 3)

    result = np.concatenate([img, warped], axis=0)

    plt.imshow(result)
    plt.show()
