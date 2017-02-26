import abc
import numpy as np
from skimage.feature import hog
from typing import Iterable
import util
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class ImageFeatureExtractorBase(abc.ABC):
    @abc.abstractmethod
    def extract(self, img: np.array) -> np.array:
        pass


class HogFeatureExtractor(ImageFeatureExtractorBase):
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        self.__cells_per_block = cells_per_block
        self.__pixels_per_cell = pixels_per_cell
        self.__orientations = orientations

    def extract(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return hog(grayscale, self.__orientations, self.__pixels_per_cell, self.__cells_per_block)


class HistFeatureExtractor(ImageFeatureExtractorBase):
    def extract(self, img: np.array) -> np.array:
        rhist = np.histogram(img[:, :, 0], bins=self.__bins, range=(0.0, 1.0))[0]
        ghist = np.histogram(img[:, :, 1], bins=self.__bins, range=(0.0, 1.0))[0]
        bhist = np.histogram(img[:, :, 2], bins=self.__bins, range=(0.0, 1.0))[0]

        return np.concatenate([rhist, ghist, bhist]).astype(np.float32)

    def __init__(self, bins=32, colorspace="RGB"):
        self.__colorspace = colorspace
        self.__bins = bins


class SpatialBinFeatureExtractor(ImageFeatureExtractorBase):
    def extract(self, img: np.array) -> np.array:
        return cv2.resize(img, (self.__dimension, self.__dimension)).ravel()

    def __init__(self, dimension=32, colorspace="RGB"):
        self.__colorspace = colorspace
        self.__dimension = dimension


class CombiningImageFeatureExtractor(ImageFeatureExtractorBase):
    def __init__(self, extractors: Iterable[ImageFeatureExtractorBase]):
        self.__extractors = extractors

    def extract(self, img):
        features = list(map(lambda e: e.extract(img), self.__extractors))
        return np.concatenate(features)


if __name__ == "__main__":
    img = util.load_image_float("dataset/train/vehicle2.png")
    plt.imshow(img)
    plt.show()
    extractor = CombiningImageFeatureExtractor([SpatialBinFeatureExtractor()])
    print(extractor.extract(img))


def load_images_and_extract(feature_extractor, img_paths):
    result_features = []
    for img_path in img_paths:
        img = util.load_image_float(img_path)
        features = feature_extractor.extract(img)
        result_features.append(features)

    result = np.array(result_features)

    return result
