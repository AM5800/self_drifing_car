import abc
import numpy as np
from skimage.feature import hog
from typing import Iterable
import util


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
        return hog(img, self.__orientations, self.__pixels_per_cell, self.__cells_per_block)


class ImageFeatureExtractor(ImageFeatureExtractorBase):
    def __init__(self, extractors: Iterable[ImageFeatureExtractorBase]):
        self.__extractors = extractors

    def extract(self, img):
        features = list(map(lambda e: e.extract(img), self.__extractors))
        return np.concatenate(features)


if __name__ == "__main__":
    img = util.load_image_float("dataset/train/vehicle1.png")[:, :, 0]
    extractor = ImageFeatureExtractor([HogFeatureExtractor()])
    print(extractor.extract(img).shape)
