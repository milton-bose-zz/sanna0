from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

#import numpy as np
#import theano
#from theano import tensor as T
import copy

from .utils import initialize_class_weights
from ..common.random import numpy_rng_instance, theano_rng_instance
from ..utils.data_processing import randomized_data_index, split_dataset
from ..utils.model_compilers import compile_model

class AdaBoostM2(object):

    def __init__(self, data, class_weight=None,
            training_fraction=0.8, numpy_rng=None, theano_rng=None,
            **kwargs):

        self.kwargs = kwargs
        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        self.data = split_dataset(data, train_fraction=training_fraction)

        self.class_weight=class_weight

        keys = ['train', 'valid']
        self._weights = {}
        for k in keys:
            self._weights[k] = initialize_class_weights(
                    self.data[k][1], class_weight=class_weight,
                    return_numpy_object=True
                    )

        self.kwargs['data'] = self.data['train']
        self.kwargs['numpy_rng'] = self.numpy_rng
        self.kwargs['theano_rng'] = self.theano_rng

        def spawn_a_model(model_kwargs):
            return compile_model(**model_kwargs)
        self.__model = spawn_a_model

    def bootstrapped_data(self):

        data = {}
        for k, v in self.data.items():
            len_ = len(v[1])
            p = self._weights[k].sum(axis=-1)
            idx = randomized_data_index(len_, size=None, replace=True,
                    p=p, numpy_rng=self.numpy_rng)
            data[k] = (v[0][idx], v[1][idx])

        return data

    def train_a_model(self, improvement_threshold=0.995,
            min_iter=2000, min_iter_increase=2, n_epochs=20):

        data = self.bootstrapped_data()
        model = self.__model(copy.deepcopy(self.kwargs))

        model.optimize_params(
                data,
                improvement_threshold=improvement_threshold,
                min_iter=min_iter,
                min_iter_increase=min_iter_increase,
                n_epochs=n_epochs
                )
        return model

