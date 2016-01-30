from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

import numpy as np

import copy

from ..common.random import numpy_rng_instance, theano_rng_instance
from ..utils.data_processing import (split_dataset, randomized_data_index)
from ..utils.model_compilers import compile_model

class Ensemble(object):

    def __init__(self, data, training_fraction=0.8,
            numpy_rng=None, theano_rng=None):

        self._keys = ['train', 'valid']

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        self.data = split_dataset(data, training_fraction=training_fraction,
                numpy_rng=self.numpy_rng)
        log_distributions(self.data['train'][1], type_='training data')
        log_distributions(self.data['valid'][1], type_='validation data')

        self._weights = {'train': None, 'valid': None}
        self.kwargs = None

        self.models = []

    def __spawn_a_model(self):
        return compile_model(**copy.deepcopy(self.kwargs))

    def bootstrapped_data(self):

        data = {}
        for k, v in self.data.items():
            len_ = len(v[1])
            if self._weights[k].ndim > 1:
                p = self._weights[k].sum(axis=-1)
            else:
                p = self._weights[k]
            idx = randomized_data_index(len_, size=None, replace=True,
                    p=p, numpy_rng=self.numpy_rng)
            data[k] = (v[0][idx], v[1][idx])
            logger.info('bootstrapped {} samples'.format(k))

        return data

    def train_a_model(self, improvement_threshold=0.995,
            min_iter=2000, min_iter_increase=2, n_epochs=20):

        data = self.bootstrapped_data()

        log_distributions(data['train'][1], type_='training sample')
        log_distributions(data['valid'][1], type_='validation sample')

        model = self.__spawn_a_model()
        model.optimize_params(
                data,
                improvement_threshold=improvement_threshold,
                min_iter=min_iter,
                min_iter_increase=min_iter_increase,
                n_epochs=n_epochs
                )
        return model

    def confidence(self, X, key='valid'):

        Y = np.zeros((len(X), 1))
        for m in self.models:
            Y = Y + m.confidence(X)

        Y = Y / Y.sum(axis=1, keepdims=True)
        return Y

    def predict(self, X, key='valid'):

        Y = self.confidence(X, key)
        return Y.argmax(axis=1)

def log_distributions(data, type_='training sample'):

    count = np.unique(data, return_counts=True)[1]
    logger.info('{0} distribution: {1}'.format(type_, count))

