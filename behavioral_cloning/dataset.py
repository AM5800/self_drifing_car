import cv2
import numpy as np


def convert_dataset(convert_fun, dataset):
    return np.array(list(map(convert_fun, dataset)))


class Dataset:
    def __init__(self, x_train, y_train, x_validation, y_validation, x_test, y_test):
        self.X_train = x_train
        self.y_train = y_train
        self.X_validation = x_validation
        self.y_validation = y_validation
        self.X_test = x_test
        self.y_test = y_test


class DatasetProvider:
    def __init__(self):
        __dataset = None
        self.__datasets = {}
        self.__converters = {}

    def initialize(self, dataset):
        self.__dataset = dataset

    def register(self, name, conversion_function):
        self.__converters[name] = conversion_function

    def get(self, name):
        if name in self.__datasets:
            return self.__datasets[name]

        conversion_function = self.__converters[name]
        origin = self.__dataset
        X_train = convert_dataset(conversion_function, origin.X_train)
        X_valid = convert_dataset(conversion_function, origin.X_validation)
        X_test = convert_dataset(conversion_function, origin.X_test)

        dataset = Dataset(X_train, origin.y_train, X_valid, origin.y_validation, X_test, origin.y_test)
        self.__datasets[name] = dataset

        return dataset

    def convert(self, name, data):
        converter = self.__converters[name]
        return convert_dataset(converter, data)

    def get_shape(self, name):
        self.get(name)
        return self.__datasets[name].X_train[0].shape

    def get_dataset_names(self):
        return self.__datasets.keys()


dataset_provider = DatasetProvider()


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


dataset_provider.register("hsv", to_hsv)
dataset_provider.register("original", lambda x: x)
