import numpy as np
import pickle
import os

def convert_dataset(convert_fun, dataset):
    return np.array(list(map(convert_fun, dataset)))


class Dataset:
    def __init__(self, x_train, y_train, x_validation, y_validation, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.x_test = x_test
        self.y_test = y_test


class DatasetProvider:
    __datasets = {}
    __converters = {}

    def __init__(self, create_dataset_function):
        file_name = "dataset.p"

        if os.path.exists(file_name):
            self.dataset = pickle.load(open(file_name, "rb"))
        else:
            self.dataset = create_dataset_function()
            pickle.dump(self.dataset, open(file_name, "wb"))

        self.register("original", lambda x: x)

    def register(self, name, conversion_function):
        origin = self.dataset
        x_train = convert_dataset(conversion_function, origin.x_train)
        x_valid = convert_dataset(conversion_function, origin.x_validation)
        x_test = convert_dataset(conversion_function, origin.x_test)

        dataset = Dataset(x_train, origin.y_train, x_valid, origin.y_validation, x_test, origin.y_test)
        self.__converters[name] = conversion_function
        self.__datasets[name] = dataset

    def get(self, name):
        return self.__datasets[name]

    def convert(self, name, data):
        converter = self.__converters[name]
        return convert_dataset(converter, data)

    def get_shape(self, name):
        return self.__datasets[name].train[0].shape

    def get_dataset_names(self):
        return self.__datasets.keys()

