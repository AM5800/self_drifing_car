import csv

import matplotlib.image as mpimg
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

import dataset
import utils


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

input_shape = dataset_provider.get_shape("original")
dataset = dataset_provider.get("original")

print("Trainset shape: ", dataset.x_train.shape)

model = Sequential([
    Conv2D(64, 3, 3, activation="relu", input_shape=input_shape),
    MaxPooling2D(),
    Conv2D(128, 3, 3, activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(1)
])


for l in model.layers:
    print(l.name, l.input_shape, l.output_shape)

model.compile(loss='mse',
              optimizer='adam')

history = model.fit(dataset.x_train, dataset.y_train,
                    batch_size=50, nb_epoch=10,
                    verbose=2, validation_data=(dataset.x_validation, dataset.y_validation))


json = model.to_json()
with open("model.json", "w") as file:
    file.write(json)

model.save_weights("model.h5")

