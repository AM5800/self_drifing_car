import util
import glob
import numpy as np
import os
from scipy.misc import imsave
import scipy
import shutil


def load_images(path):
    result = []
    path = os.path.join(path, "**/*")
    for file in glob.glob(path, recursive=True):
        img = util.try_load_image_float(file)
        if img is not None:
            result.append(img)
    return np.array(result)


def split_to_dataset(input, desired_validation_size, desired_test_size):
    validation = input[:desired_validation_size]
    test = input[desired_validation_size:desired_test_size + desired_validation_size]
    train = input[desired_test_size + desired_validation_size:]
    return train, validation, test


def save(images, path, file_mask):
    i = 0

    if not os.path.exists(path):
        os.makedirs(path)

    for img in images:
        while True:
            i += 1
            img_path = os.path.join(path, file_mask.format(i))
            if not os.path.isfile(img_path):
                scipy.misc.imsave(img_path, img)
                break


def prepare_dataset(vehicles_path, non_vehicles_path, output_path):
    vehicles = load_images(vehicles_path)
    non_vehicles = load_images(non_vehicles_path)

    np.random.shuffle(vehicles)
    np.random.shuffle(non_vehicles)

    desired_validation_size = 1000
    desired_test_size = 1000

    v_train, v_valid, v_test = split_to_dataset(vehicles, desired_validation_size, desired_test_size)
    nv_train, nv_valid, nv_test = split_to_dataset(non_vehicles, desired_validation_size, desired_test_size)

    train_out = os.path.join(output_path, "train")
    valid_out = os.path.join(output_path, "valid")
    test_out = os.path.join(output_path, "test")

    save(v_train, train_out, "vehicle{0}.png")
    save(nv_train, train_out, "non-vehicle{0}.png")

    save(v_valid, valid_out, "vehicle{0}.png")
    save(nv_valid, valid_out, "non-vehicle{0}.png")

    save(v_test, test_out, "vehicle{0}.png")
    save(nv_test, test_out, "non-vehicle{0}.png")


if __name__ == "__main__":
    out_dir = "dataset"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    prepare_dataset("vehicles", "non-vehicles", out_dir)
