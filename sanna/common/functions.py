# -*- coding: utf-8 -*-
from __future__ import print_function, division

from theano import tensor as T


def softplus(X):

    return T.nnet.softplus(X)


def linear(X, M, b):

    return T.dot(X, M) + b


def sigmoid(X):

    return T.nnet.sigmoid(X)


def softmax(X):

    return T.nnet.softmax(X)


def tanh(X):

    return T.tanh(X)


def margin(Y, y):

    return Y * y


def SE(Y, y):

    return ((y - Y) ** 2).mean(axis=0) / 2


def AE(Y, y):

    return abs(Y - y).mean(axis=0)


def negative_log_proba(Y, y):

    return - T.log(Y)[T.arange(y.shape[0]), y]


def zero_one_loss(Y, y):

    Y = T.argmax(Y, axis=1)
    return T.neq(Y, y)


def argmax(Y):

    return T.argmax(Y, axis=-1)
