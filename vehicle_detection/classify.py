import glob
import os
import pickle
import time
from feature_extractor import *
from vehicle_classifier import *


def load_dataset_paths(path):
    limit = 50000
    vehicles = glob.glob(os.path.join(path, "vehicle*"))[:limit]
    non_vehicles = glob.glob(os.path.join(path, "non-vehicle*"))[:limit]

    vehicle_labels = np.array(list(map(lambda x: 0, vehicles)))
    non_vehicle_labels = np.array(list(map(lambda x: 1, non_vehicles)))

    paths = np.concatenate([vehicles, non_vehicles])
    labels = np.concatenate([vehicle_labels, non_vehicle_labels])

    return paths, labels


def compute_accuracy(model: VehicleClassifierInterface, xs, ys):
    prediction = model.predict(xs)
    wrongs = np.count_nonzero(ys - prediction)
    total = len(ys)
    return (total - wrongs) / total


if __name__ == "__main__":
    hog_extractor = HogFeatureExtractor(9, (8, 8), (2, 2))
    hist_extractor = HistFeatureExtractor(16, "HSV")
    sb_extractor = SpatialBinFeatureExtractor(16)

    extractor = CombiningImageFeatureExtractor([hog_extractor, hist_extractor, sb_extractor])

    train_X, train_y = load_dataset_paths("dataset/train")
    train_X, train_y = util.parallel_shuffle([train_X, train_y])

    valid_X, valid_y = load_dataset_paths("dataset/valid")

    print("Training...")
    t0 = time.time()
    classifier = SVMVehicleClassifier(extractor)
    classifier.fit(train_X, train_y)

    pickle.dump(classifier, open("classifier.p", "wb"))

    print("Elapsed:", time.time() - t0)
    print("Accuracy:", compute_accuracy(classifier, valid_X, valid_y))