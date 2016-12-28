import numpy as np
import dataset


def shuffle_split_to_dataset(xs, ys, train_ratio, validation_ratio):
    if len(xs) != len(ys):
        raise Exception("X and Y length differs")

    total_len = len(ys)
    train_len = int(train_ratio * total_len)
    valid_len = int(validation_ratio * total_len)

    xs, ys = parallel_shuffle(xs, ys)

    x_train = xs[:train_len]
    y_train = ys[:train_len]

    x_valid = xs[train_len:valid_len + train_len]
    y_valid = ys[train_len:valid_len + train_len]

    x_test = xs[train_len + valid_len:]
    y_test = ys[train_len + valid_len:]

    return dataset.Dataset(x_train, y_train, x_valid, y_valid, x_test, y_test)


def parallel_shuffle(*args):
    result = []
    length = len(args[0])
    permutations = np.random.permutation(length)
    for arg in args:
        if len(arg) != length:
            raise Exception("All lists should have the same length")

        result.append(arg[permutations])

    return result
