import csv
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Sequential
import os
import matplotlib.image as mpimg
import copy
from keras import backend as kb
import time
from dataset import dataset_provider, Dataset


# Reads driving log and loads center image + steering angle
def read_log_file(file_path):
    xs = []
    ys = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            img_path = row[0]

            if not os.path.isfile(img_path):
                continue

            steering_angle = float(row[3])

            img = mpimg.imread(img_path)
            xs.append(img)
            ys.append(steering_angle)

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys


def load_dataset():
    # Train set and validation set both were created manually
    # and held separately
    X_train, y_train = read_log_file("train/driving_log.csv")
    X_valid, y_valid = read_log_file("validation/driving_log.csv")

    return Dataset(X_train, y_train, X_valid, y_valid, [], [])


dataset_provider.initialize(load_dataset())


def save_model(model, dataset_name, name):
    json = model.to_json()
    with open(name + ".json", "w") as file:
        file.write(json)

    model.save_weights(name + ".h5")
    with open(name + ".cfg", "w") as cfg:
        cfg.write(dataset_name)


# Defines AlexNet-like network
# With 4 Convolution layers (Conv2d -> BN -> relu -> MaxPool)
# And 2 Fully-connected layers
def alexnet(input_shape, dropout, use_bn):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=input_shape))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, 3))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, 3))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, 3, 3))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(1000))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(Dropout(dropout))

    model.add(Dense(1))

    return model


# Defines VGG-like network
# With 9 Convolution layers (Conv2d -> relu -> BN -> MaxPool)
# And 3 Fully-connected layers
def vgg(input_shape, dropout, use_bn):
    model = Sequential()

    model.add(Conv2D(64, 3, 3, activation="relu", input_shape=input_shape))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Conv2D(64, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Conv2D(128, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Conv2D(128, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Conv2D(256, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Conv2D(256, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Conv2D(512, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Conv2D(512, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Conv2D(1024, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Conv2D(1024, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(4096, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Dense(1000, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Dropout(dropout))

    model.add(Dense(1))

    return model


grid = [{}]


# Computes cartesian product of values and grid content
def add_to_grid(name, values):
    global grid

    new_grid = []
    for node in grid:
        for value in values:
            node_copy = copy.deepcopy(node)
            node_copy[name] = value
            new_grid.append(node_copy)

    grid = new_grid


add_to_grid("lr", [0.0001, 0.00001, 0.001])
add_to_grid("dropout", [1.0, 0.5, 0.7, 0.3])
add_to_grid("bn", [True, False])
add_to_grid("dataset", ["original", "hsv"])
add_to_grid("F", [vgg, alexnet])

best_val_loss = 100
# Grid node with the least validation loss will be saved in this variable
best_node = None

for i in range(len(grid)):
    node = grid[i]
    node["grid"] = i + 1

    print(node)

    lr = node["lr"]
    dropout = node["dropout"]
    bn = node["bn"]
    dataset_name = node["dataset"]
    F = node["F"]

    dataset = dataset_provider.get(dataset_name)
    dataset_shape = dataset_provider.get_shape(dataset_name)

    model = F(dataset_shape, dropout, bn)

    model.compile(loss='mse',
                  optimizer='adam',
                  lr=lr)

    t0 = time.time()

    model.fit(dataset.X_train, dataset.y_train,
              batch_size=10, nb_epoch=1,
              verbose=0)

    val_loss = model.evaluate(dataset.X_validation, dataset.y_validation, verbose=0)

    elapsed_time = time.time() - t0

    node["val_loss"] = val_loss
    node["elapsed_time"] = elapsed_time

    print("val_loss", val_loss)
    print("elapsed time:", elapsed_time)
    print()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_node = node
        save_model(model, dataset_name, "model")

    # Clear session to avoid OOM in very big grids
    kb.clear_session()

print("Best node:", best_node)
