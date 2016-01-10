# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
from theano import tensor as T

from .functions import *

def mean_kappa_loss(Y, y, y_distro=None, power=2, class_weights=None,
                    n_classes=None):

    if y_distro is None:
        y_distro = np.ones(n_classes) / n_classes

    actual_distro = T.constant(
            np.asarray(y_distro)[:, None]
            )
    ycols = Y.shape[1]
    a = T.arange(ycols)
    w_num = a.dimshuffle('x', 0) - y .dimshuffle(0, 'x')
    w = a.dimshuffle(0, 'x') - a.dimshuffle('x', 0)
    if power == 1:
        w = abs(w)
        w_num = abs(w_num)
    else:
        w = w ** power
        w_num = w_num ** power

    num = T.sum(Y * w_num, axis = -1)
    den = T.dot(T.dot(Y, w), actual_distro).ravel()

    m = T.sum(num / den)
    if class_weights is None:
        return T.mean(m)
    else:
        return T.mean(class_weights[y] * m)


def mean_weighted_loss(Y, y, power=1, class_weights=None):

    ycols = Y.shape[-1]
    a = T.arange(ycols)[None, :]
    w = a - y.dimshuffle(0, 'x')
    if power == 1:
        w = abs(w)
    else:
        w = w ** power
    m = T.sum(Y * w, axis=-1)
    if class_weights is None:
        return T.mean(m)
    else:
        class_weights = np.asarray(class_weights)
        class_weights = T.constant(class_weights)
        return T.mean(class_weights[y] * m)


def MSE(Y, y):

    return T.mean(SE(Y, y))


def MAE(Y, y):

    return T.mean(AE(Y, y))


def mean_neg_log_p(Y, y, class_weights=None):

    m = negative_log_proba(Y, y)
    if class_weights is None:
        return T.mean(m)
    else:
        class_weights = np.asarray(class_weights)
        class_weights = T.constant(class_weights)
        return T.mean(class_weights[y] * m)


def mean_zero_one_loss(Y, y):

    return T.mean(zero_one_loss(Y, y))


def tot_zero_one_loss(Y, y):

    return T.sum(zero_one_loss(Y, y))


def mean_margin_loss(Y, y):

    return - T.mean(margin(Y, y))


def adaboost_loss(Y, y, alpha=1):

    return T.mean(T.exp(- alpha * margin(Y, y)))
