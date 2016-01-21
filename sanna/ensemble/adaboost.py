from __future__ import print_funtion, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
import theano
from theano import tensor as T

from .utils import initialize_class_weights

class AdaBoostM2(object):

    def __init__(self, model_class, data, class_weight=None,
            training_fraction=0.8, **kwargs):

        self.model_class = model_class
        self.class_weight=class_weight

        len_ = len(data[0])
        size = int(len_ * training_fraction)
        self.data = dict(
                train=(data[0][:size], data[1][:size]),
                valid=(data[0][size:], data[1][size:])
                )

        keys = ['train', 'valid']
        self._weights = {}
        for k in keys:
            self._weights[k] = initialize_class_weights(
                    self.data[k][1], class_weight=class_weight,
                    return_numpy_object=True
                    )

        def spawn_a_model(self):
            return model_class(**kwargs)





