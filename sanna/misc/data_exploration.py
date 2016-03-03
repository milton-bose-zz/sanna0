# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np


def has_missing_values(X, axis=0):

    pos = np.argwhere(np.any(np.isnan(X), axis=axis)).flatten()
    return pos


def integer_columns(X):
    dct = {}
    for i in np.arange(X.shape[1]):
        arr = np.unique(X[:, i])
        arr = arr[~np.isnan(arr)]
        diff = np.nan_to_num(arr - np.round(arr))
        if np.sum(diff) == 0:
            dct[i] = arr
    return dct


def get_integer_columns(X):

    cols = integer_columns(X).keys()
    cols = sorted([c for c in cols])
    return np.array(cols)


def get_non_int_columns(X):

    cols = np.arange(X.shape[1])
    int_cols = get_integer_columns(X)
    non_int_cols = cols[~np.in1d(cols, int_cols)]
    return non_int_cols


def seperate_int_features(X):

    cols = np.arange(X.shape[1])
    int_cols = get_integer_columns(X)
    X_int = X[:, int_cols]
    X_non_int = X[:, ~np.in1d(cols, int_cols)]
    return X_int, X_non_int
