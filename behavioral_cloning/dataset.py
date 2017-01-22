import cv2
import numpy as np
import matplotlib.image as mpimg
import queue


def parallel_shuffle(a, b):
    if len(a) != len(b):
        raise Exception("arguments size mismatch")
    permutation = np.random.permutation(len(a))

    return a[permutation], b[permutation]


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
        self.__dataset = None
        self.__converters = {}

    def initialize(self, dataset):
        self.__dataset = dataset

    def register(self, name, conversion_function):
        self.__converters[name] = conversion_function

    def convert(self, name, data):
        converter = self.__converters[name]
        return convert_dataset(converter, data)

    def get_shape(self, name):
        sample = self.__dataset.X_train[0]
        img = mpimg.imread(sample)
        converted = self.__converters[name](img)
        return converted.shape

    def get_dataset_names(self):
        return self.__converters.keys()

    def get_train_generator(self, dataset_name, batch_size):
        return self.__generator(self.__dataset.X_train, self.__dataset.y_train, dataset_name, batch_size)

    def get_valid_generator(self, dataset_name, batch_size):
        return self.__generator(self.__dataset.X_validation, self.__dataset.y_validation, dataset_name, batch_size)

    def __generator(self, xs, ys, dataset_name, batch_size):
        q = queue.Queue()
        while True:

            xs, ys = parallel_shuffle(xs, ys)

            for i in range(len(xs)):
                img_path = xs[i]
                y = ys[i]
                img = mpimg.imread(img_path)
                img = self.__converters[dataset_name](img)

                q.put((img, y))
                if q.qsize() >= batch_size or i == len(xs) - 1:
                    X_batch = np.array(list(x[0] for x in q.queue))
                    y_batch = np.array(list(x[1] for x in q.queue))
                    q.queue.clear()
                    yield (X_batch, y_batch)

    def get_train_size(self):
        return len(self.__dataset.X_train)

    def get_val_size(self):
        return len(self.__dataset.X_validation)


dataset_provider = DatasetProvider()


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def to_norm_rgb(img):
    return img / 255.0 - 0.5


dataset_provider.register("hsv", to_hsv)
dataset_provider.register("original", lambda x: x)
dataset_provider.register("norm_rgb", to_norm_rgb)
