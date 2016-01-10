from __future__ import print_function, division

import numpy as np
import theano
from theano import shared

from ..common.random import numpy_rng_instance


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


def randomized_data_index(data_len, size=None, replace=True, p=None,
                          numpy_rng=None):
    if size is None:
        size = data_len

    numpy_rng = numpy_rng_instance(numpy_rng)
    idx = np.arange(data_len)
    idx = numpy_rng.choice(idx, size=size, replace=replace, p=p)

    return idx


def split_dataset(data, train_fraction=0.70, n_ycols=1,
                  shuffle=True, numpy_rng=None, flatten_y=True):
    """Split the dataset

    """

    numpy_rng = numpy_rng_instance(numpy_rng)
    len_ = len(data)
    train_size = int(train_fraction * len_)

    if shuffle:
        numpy_rng.shuffle(data)

    train = data[:train_size]
    valid = data[train_size:]

    if flatten_y:
        train_xy = (
            train[:, :-n_ycols],
            train[:, -n_ycols:].ravel()
        )
        valid_xy = (
            valid[:, :-n_ycols],
            valid[:, -n_ycols:].ravel()
        )
    else:
        train_xy = (
            train[:, :-n_ycols],
            train[:, -n_ycols:]
        )
        valid_xy = (
            valid[:, :-n_ycols],
            valid[:, -n_ycols:]
        )

    return (train_xy, valid_xy)


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
