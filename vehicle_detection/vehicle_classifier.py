import abc

from sklearn import svm
from sklearn.preprocessing import StandardScaler

import feature_extractor


class VehicleClassifierInterface(abc.ABC):
    @abc.abstractmethod
    def predict(self, features):
        pass

    @abc.abstractmethod
    def fit(self, features, labels):
        pass


class SVMVehicleClassifier(VehicleClassifierInterface):
    def __init__(self, feature_extractor: feature_extractor.ImageFeatureExtractorInterface):
        self.__feature_extractor = feature_extractor
        self.__svc = svm.SVC()
        self.__scaler = None

    def fit(self, features, labels):
        features = feature_extractor.load_images_and_extract(self.__feature_extractor, features)

        self.__scaler = StandardScaler().fit(features)
        features = self.__scaler.transform(features)

        self.__svc.fit(features, labels)

    def predict(self, features):
        features = feature_extractor.load_images_and_extract(self.__feature_extractor, features)
        features = self.__scaler.transform(features)
        return self.__svc.predict(features)
