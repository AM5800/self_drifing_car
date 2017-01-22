import csv
import os
import time

import keras.applications
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as kb
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D
from keras.models import Sequential, Model
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

            xs.append(img_path)
            ys.append(steering_angle)

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys


def load_dataset():
    # Train set and validation set both were created manually
    # and held separately
    train_files = ["train/t1_center_2l/driving_log.csv",
                   "train/t1_recovery/driving_log.csv",
                   "train/t2_center_0l/driving_log.csv"]

    valid_files = ["validation/iteration1/driving_log.csv",
                   "validation/t2_center_0l/driving_log.csv",
                   "validation/t2_recovery/driving_log.csv"]
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []

    for log in train_files:
        xs, ys = read_log_file(log)
        X_train.extend(xs)
        y_train.extend(ys)

    for log in valid_files:
        xs, ys = read_log_file(log)
        X_valid.extend(xs)
        y_valid.extend(ys)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    return Dataset(X_train, y_train, X_valid, y_valid, [], [])


dataset_provider.initialize(load_dataset())


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

    if dropout is not None:
        model.add(Dropout(dropout))

    model.add(Dense(1))

    return model


def inceptionv3(input_shape, dropout, use_bn):
    if not use_bn:
        return None
    if dropout is not None:
        return None

    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                             input_shape=input_shape)

    output = base_model.output
    avg = AveragePooling2D(pool_size=(3, 8))(output)
    flatten = Flatten()(avg)
    dense1 = Dense(1024, activation="relu")(flatten)
    dense2 = Dense(1)(dense1)

    model = Model(input=base_model.input, output=dense2)

    return model


models = {"alexnet": alexnet, "inceptionv3": inceptionv3}

grid_manager = grid.GridManager("history.p")
grid_manager.add("lr", [None])
grid_manager.add("model", ["alexnet", "inceptionv3"])
grid_manager.add("dropout", [None, 0.5, 0.3, 0.7, 0.65, 0.75, 0.0])
grid_manager.add("bn", [True, False])
grid_manager.add("dataset", ["norm_rgb", "original", "hsv", ])


class GlobalModelCheckpoint(Callback):
    def __init__(self, node):
        super().__init__()
        self.__node = node

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs["val_loss"]

        if epoch > 1 and val_loss < grid_manager.get_best_result_value():
            print("New best model:", self.__node)
            print("val_loss:", val_loss)
            self.save_model(self.model, self.__node["dataset"], "model")
            plot(model, to_file='model.png', show_layer_names=False, show_shapes=True)
            grid_manager.submit_best_result_value(val_loss)

    @staticmethod
    def save_model(model, dataset_name, name):
        json = model.to_json()
        with open(name + ".json", "w") as file:
            file.write(json)

        model.save_weights(name + ".h5")
        with open(name + ".cfg", "w") as cfg:
            cfg.write(dataset_name)


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

    if model is None:
        continue

    model.compile(loss='mse',
                  optimizer='adam',
                  lr=lr)

    t0 = time.time()

    train_size = dataset_provider.get_train_size()
    val_size = dataset_provider.get_val_size()

    es = EarlyStopping(patience=5, min_delta=0.01)
    saver = GlobalModelCheckpoint(node)

    batch_size = 10
    train_generator = dataset_provider.get_train_generator(dataset_name, batch_size)
    valid_generator = dataset_provider.get_valid_generator(dataset_name, batch_size)

    history = model.fit_generator(train_generator, samples_per_epoch=train_size, callbacks=[es, saver],
                                  nb_epoch=50, validation_data=valid_generator, nb_val_samples=val_size,
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
    print("Model parameters:", model.count_params())
    print()

    grid_manager.save_node_result(node, result)

    # Clear session to avoid OOM in very big grids
    kb.clear_session()


def topN(filter_name, n):
    results = grid_manager.get_results()
    filtered = filter(lambda x: x[0]["model"] == filter_name, results)
    return list(sorted(filtered, key=lambda r: r[1]["val_loss"])[:n])


top_alexnet = topN("alexnet", 3)
top_inception = topN("inceptionv3", 3)
handles = []
top = top_alexnet + top_inception
for i in range(len(top)):
    node = top[i][0]
    result = top[i][1]
    label = "Model {0}".format(i)
    h = plt.plot(result["val_loss_history"], label=label)
    handles.append(h[0])
    print("{0} = {1}".format(label, node))
    print("val_loss: {0}, elapsed: {1}, epochs: {2}".format(result["val_loss"],
                                                            result["elapsed_time"],
                                                            len(result["val_loss_history"])))
    print()

plt.legend(handles=handles)
plt.ylabel("val_loss")
plt.ylabel("epoch")
plt.show()
