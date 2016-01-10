# -*- coding: utf-8 -*-
from __future__ import print_function, division

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set_context('poster')


def train_validation_trace(training_loss, validation_loss,
                           left_scales=['log', 'log'],
                           right_scales=['linear', 'log'],
                           figsize=(12, 5), i=1):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[0].set(xscale=left_scales[0], yscale=left_scales[1])
    ax[0].plot(training_loss, validation_loss, marker='o')
    ax[0].plot(training_loss[i-1: i], validation_loss[i-1: i],
               marker='o', color='red')
    ax[0].plot(ax[0].get_xlim(), ax[0].get_xlim(), ls="--", c=".3")

    ax[0].set_xlabel('training loss')
    ax[0].set_ylabel('validation loss')

    idx = np.arange(len(training_loss))
    ax[1].set(xscale=right_scales[0], yscale=right_scales[1])
    ax[1].plot(idx, training_loss, marker='o', label='training loss')
    ax[1].plot(idx, validation_loss, marker='o', label='validation loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    return fig
