import csv
import os
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as kb
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils.visualize_util import plot

import grid
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
            xs.append(img_path)
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
# With 4 Convolution layers (Conv2d -> relu -> BN -> MaxPool)
# And 3 Fully-connected layers
def alexnet(input_shape, dropout, use_bn):
    model = Sequential()

    model.add(Conv2D(32, 3, 3, activation="relu", input_shape=input_shape))
    if use_bn:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(128, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(256, 3, 3, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(1024, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Dense(256, activation="relu"))
    if use_bn:
        model.add(BatchNormalization())

    model.add(Dropout(dropout))

    model.add(Dense(1))

    return model


models = {"alexnet": alexnet}

grid_manager = grid.GridManager("history.p")
grid_manager.add("lr", [None, 0.0001, 0.01, 0.00001])
grid_manager.add("model", ["alexnet"])
grid_manager.add("dropout", [0.5, 0.3, 0.7, 0.65, 0.75, 0.0])
grid_manager.add("bn", [True, False])
grid_manager.add("dataset", ["norm_rgb", "original", "hsv", ])


def image_loader(input):
    img_path = input[0]
    steering_angle = input[1]
    img = mpimg.imread(img_path)
    return img, steering_angle


new_nodes = grid_manager.get_new_nodes()
for i in range(len(new_nodes)):
    node = new_nodes[i]

    print("{0}/{1}".format(i + 1, len(new_nodes)))
    print(node)

    lr = node["lr"]
    dropout = node["dropout"]
    bn = node["bn"]
    dataset_name = node["dataset"]
    net = models[node["model"]]

    dataset_shape = dataset_provider.get_shape(dataset_name)

    model = net(dataset_shape, dropout, bn)

    model.compile(loss='mse',
                  optimizer='adam',
                  lr=lr)

    t0 = time.time()

    train_size = dataset_provider.get_train_size()
    val_size = dataset_provider.get_val_size()

    es = EarlyStopping(patience=3, min_delta=30.0 / train_size)

    batch_size = 10
    train_generator = dataset_provider.get_train_generator(dataset_name, batch_size)
    valid_generator = dataset_provider.get_valid_generator(dataset_name, batch_size)

    history = model.fit_generator(train_generator, samples_per_epoch=train_size, callbacks=[es],
                                  nb_epoch=30, validation_data=valid_generator, nb_val_samples=val_size,
                                  verbose=0)

    elapsed_time = time.time() - t0

    result = {}

    val_loss = history.history["val_loss"][-1]
    result["elapsed_time"] = elapsed_time
    result["val_loss_history"] = history.history["val_loss"]
    result["loss_history"] = history.history["loss"]
    result["val_loss"] = val_loss

    print("val_loss", val_loss)
    print("elapsed time:", elapsed_time)
    print()

    if val_loss < grid_manager.get_best_result_value():
        best_val_loss = val_loss
        save_model(model, dataset_name, "model")
        plot(model, to_file='model.png', show_layer_names=False, show_shapes=True)

    grid_manager.submit(val_loss, node, result)

    # Clear session to avoid OOM in very big grids
    kb.clear_session()

top = sorted(grid_manager.get_results(), key=lambda r: r[1]["val_loss"])[:5]
handles = []
for i in range(len(top)):
    node = top[i][0]
    result = top[i][1]
    label = "Model {0}".format(i)
    h = plt.plot(result["val_loss_history"], label=label)
    handles.append(h[0])
    print("{0} = {1}".format(label, node))
    print("val_loss: {0}, elapsed: {1}, epochs: {2}".format(result["val_loss"], result["elapsed_time"],
                                                            len(result["val_loss_history"])))
    print()

plt.legend(handles=handles)
plt.ylabel("val_loss")
plt.ylabel("epoch")
plt.show()
