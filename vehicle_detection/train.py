import glob
import os
from feature_extractor import *
import util
from sklearn.preprocessing import StandardScaler
from sklearn import svm


def load_dataset_paths(path):
    vehicles = glob.glob(os.path.join(path, "vehicle*"))
    non_vehicles = glob.glob(os.path.join(path, "non-vehicle*"))

    vehicle_labels = np.array(list(map(lambda x: 0, vehicles)))
    non_vehicle_labels = np.array(list(map(lambda x: 1, non_vehicles)))

    paths = np.concatenate([vehicles, non_vehicles])
    labels = np.concatenate([vehicle_labels, non_vehicle_labels])

    return paths, labels


def extract_features(img_paths, feature_extractor: ImageFeatureExtractorBase, scaler: StandardScaler = None):
    result_features = []
    for img_path in img_paths:
        img = util.load_image_float(img_path)
        features = feature_extractor.extract(img)
        result_features.append(features)

    result = np.array(result_features)

    if scaler is not None:
        result = scaler.transform(result)

    return result


def evaluate_svm(model, xs, ys):
    prediction = model.predict(xs)
    accuracy = np.dot(prediction, ys)

    return accuracy / len(ys)


if __name__ == "__main__":
    extractor = HogFeatureExtractor()

    train_X, train_y = load_dataset_paths("dataset/train")
    train_X = extract_features(train_X, extractor)

    print(train_X.shape)

    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)

    train_X, train_y = util.parallel_shuffle([train_X, train_y])

    valid_X, valid_y = load_dataset_paths("dataset/valid")
    valid_X = extract_features(valid_X, extractor, scaler)

    clf = svm.SVC()
    clf.fit(train_X, train_y)

    print(evaluate_svm(clf, valid_X, valid_y))