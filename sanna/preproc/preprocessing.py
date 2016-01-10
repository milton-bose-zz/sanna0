# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
from sklearn import preprocessing as skpreproc


def binarize_ordered_class(y, include_opposite=False):
    """Binerizes an ordered classes.

    An ooredered classifier with classes 0, 1, 2 is represented as::

    0 --> [0, 0]
    1 --> [1, 0]
    2 --> [1, 1]

    if include_opposite is ``True`` then::

    0 --> [0, 0, 1, 1]
    1 --> [1, 0, 0, 1]
    2 --> [1, 1, 0, 0]


    """

    u = np.unique(y)
    lst = []
    for i in u[:-1]:
        lst.append(
            (y > i).astype(int)
        )

    if include_opposite:
        for i in u[1:]:
            lst.append(
                (y < i).astype(int)
            )

    return np.vstack(lst).T


def impute_missing_int(X_int, offset=1):

    max_ = np.nan_to_num(X_int).max()

    X_int[np.isnan(X_int)] = max_ + offset

    return X_int


class Imputer(skpreproc.Imputer):
    """Impute missing values
    """

    def __init__(self, strategy='mean', axis=0, verbose=0,
                 copy=True):
        skpreproc.Imputer.__init__(self, strategy=strategy, axis=axis,
                                   verbose=verbose, copy=copy)

    def fit(self, X, y=None):
        if self.strategy == 'new':
            self.impute_value = np.nan_to_num(X).max() + 1
            return self
        else:
            return super(Imputer, self).fit(X, y)

    def transform(self, X):
        if self.strategy == 'new':
            X[np.isnan(X)] = self.impute_value
            return X
        else:
            return super(Imputer, self).transform(X)

    def fit_transform(self, X, y=None):
        if self.strategy == 'new':
            self.impute_value = np.nan_to_num(X).max() + 1
            X[np.isnan(X)] = self.impute_value
            return X
        else:
            return super(Imputer, self).fit_transform(X, y)


class HybridTransformer(object):
    """Hybrid transformer
    """
    def __init__(self, pipelines, col_numbers):

        self.pipelines = pipelines
        self.col_numbers = col_numbers

    def fit(self, X, y=None):

        for pl_, cols_ in zip(self.pipelines, self.col_numbers):
            pl_ = pl_.fit(X[:, cols_])  # Works since classes are passed by ref

        return self

    def transform(self, X):

        X_lst = []
        for pl_, cols_ in zip(self.pipelines, self.col_numbers):
            X_lst.append(pl_.transform(X[:, cols_]))
            # print(pl_.transform(X[:, cols_]).shape)
        try:
            transformed = np.hstack(X_lst)
        except:
            print('Set the sparse property to `False`')

        return transformed

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)
