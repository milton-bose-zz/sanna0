# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np


def quadratic_weighted_kappa(prediction, actual, n_classes=None):

    if n_classes is None:
        n_classes = len(np.unique(actual))
    a = np.arange(n_classes)
    w = (a[:, None] - a[None, :]) ** 2

    return kappa(prediction, actual, n_classes=n_classes, weight_matrix=w)


def kappa(prediction, actual, n_classes=None, weight_matrix=None):

    if n_classes is None:
        n_classes = len(np.unique(actual))
    if weight_matrix is None:
        w = np.ones((n_classes, n_classes))
        np.fill_diagonal(w, 0)
    else:
        w = weight_matrix


    cm = confusion_matrix(prediction, actual, n_classes=n_classes)

    len_ = len(actual)
    actual_proba = cm.sum(axis=1) / len_
    prediction_proba = cm.sum(axis=0) / len_
    den = actual_proba[:, None] * prediction_proba[None, :] * len_

    kappa_ = 1 - (cm * w).sum() / (den * w).sum()

    return kappa_


def confusion_matrix(prediction, actual, n_classes=None, scaled=False):

    if n_classes is None:
        n_classes = len(np.unique(actual))

    bins = np.arange(n_classes + 1) - 0.5
    mat = np.histogram2d(actual, prediction, bins=bins)[0]
    mat = mat.astype(int)

    if scaled:
        mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def MSE(pred, actual):

    return np.mean((pred - actual) ** 2) / 2

def zero_one_loss(pred, actual):
    return pred != actual


def tot_zero_one_loss(pred, actual):
    return np.sum(zero_one_loss(pred, actual))


def mean_zero_one_loss(pred, actual):
    return np.mean(zero_one_loss(pred, actual))
