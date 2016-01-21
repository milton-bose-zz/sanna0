from __future__ import print_funtion, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
import theano
from theano import tensor as T

from .utils import initialize_class_weights
from ..common.random import numpy_rng_instance, theano_rng_instance
from ..utils.data_procession import randomized_data_index, split_dataset

class AdaBoostM2(object):

    def __init__(self, model_class, data, class_weight=None,
            training_fraction=0.8, numpy_rng=None, theano_rng=None,
            **kwargs):

        numpy_rng = numpy_rng_instance(numpy_rng)
        theano_rng = theano_rng_instance(theano_rng)

        self.data = split_dataset(data, train_fraction=training_fraction)


        self.model_class = model_class
        self.class_weight=class_weight

        keys = ['train', 'valid']
        self._weights = {}
        for k in keys:
            self._weights[k] = initialize_class_weights(
                    self.data[k][1], class_weight=class_weight,
                    return_numpy_object=True
                    )

        def spawn_a_model(self):
            return model_class(**kwargs)





