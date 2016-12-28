import csv
import matplotlib.image as mpimg
import numpy as np
import utils
import dataset


def load_dataset(log_file="driving_log.csv"):
    xs = []
    ys = []

    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            img_path = row[0]
            steering_angle = row[3]

            img = mpimg.imread(img_path)

            xs.append(img)
            ys.append(steering_angle)

    xs = np.array(xs)
    ys = np.array(ys)
    return utils.shuffle_split_to_dataset(xs, ys, 0.6, 0.2)


dataset_provider = dataset.DatasetProvider(load_dataset)
