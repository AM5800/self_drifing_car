import abc
from sklearn import svm
import feature_extractor
from sklearn.preprocessing import StandardScaler


class VehicleClassifier(abc.ABC):
    @abc.abstractmethod
    def predict(self, features):
        pass

    @abc.abstractmethod
    def fit(self, features, labels):
        pass


class SVMVehicleClassifier(VehicleClassifier):
    def __init__(self, feature_extractor: feature_extractor.ImageFeatureExtractorBase):
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
