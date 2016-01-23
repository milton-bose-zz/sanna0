from __future__ import print_function, division

import numpy as np
import theano
from theano import shared

from ..common.random import numpy_rng_instance

def binarize(arr, n_classes=None):

    arr = arr.astype(int)

    if n_classes is None:
        n_classes = arr.max() + 1

    output = np.zeros((len(arr), n_classes))
    output[np.arange(len(arr)), arr] = 1

    return output


def shared_dataset(data_xy, borrow=True, name='train'):

    data_x, data_y = data_xy

    shared_x = shared(
        np.asarray(data_x, dtype=theano.config.floatX),
        borrow=borrow,
        name=name+'_x')

    shared_y = shared(
        np.asarray(data_y, dtype=theano.config.floatX),
        borrow=borrow,
        name=name+'_y')

    return shared_x, shared_y


def inverse_frequency(y):

    u, n = np.unique(y, return_counts=True)
    w = np.zeros(len(y))

    for i, v in enumerate(u):
        w += (y == v)/n[i]

    return w/len(u)


def randomized_data_index(data_len, size=None, replace=True,
        p=None, numpy_rng=None):
    if size is None:
        size = data_len

    numpy_rng = numpy_rng_instance(numpy_rng)
    idx = np.arange(data_len)
    idx = numpy_rng.choice(idx, size=size, replace=replace, p=p)

    return idx


def split_dataset(data, training_fraction=0.80, shuffle=False,
        numpy_rng=None):
    """Split the dataset

    """
    assert training_fraction <= 1.0
    assert training_fraction > 0.0
    numpy_rng = numpy_rng_instance(numpy_rng)
    len_ = len(data[0])
    size = int(training_fraction * len_)

    if shuffle:
        idx = randomized_data_index(len_, size=size, replace=False,
                p=None, numpy_rng=numpy_rng)
        data = (data[0][idx], data[1][idx])

    data = dict(
            train=(data[0][:size], data[1][:size]),
            valid=(data[0][size:], data[1][size:])
            )

    return data



def load_data(datasets, borrow=True):

    train = datasets[0]
    train_x, train_y = shared_dataset(train, borrow, name='train')

    try:
        valid = datasets[1]
    except:
        pass
    else:
        valid_x, valid_y = shared_dataset(valid, borrow, name='valid')

    if 'valid' in locals():
        return [(train_x, train_y), (valid_x, valid_y)]
    else:
        return (train_x, train_y)
