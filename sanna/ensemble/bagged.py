from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
#import theano
#from theano import tensor as T
import copy

from ..common.random import numpy_rng_instance, theano_rng_instance
from ..utils.data_processing import (randomized_data_index,
                                    split_dataset)
from ..utils.model_compilers import compile_model

class Bagged(object):

    def __init__(self, data, sample_weights=None,
            training_fraction=0.8, numpy_rng=None, theano_rng=None,
            **kwargs):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        logger.info('splitting the data into `train` and `valid` sets')
        self.data = split_dataset(data, training_fraction=training_fraction,
                numpy_rng=numpy_rng)
        len_ = len(self.data['train'][0])
        self._weights = dict(
                train=sample_weights[:len_],
                valid=sample_weights[len_:]
                )

        log_distributions(self.data['train'][1], type_='training data')
        log_distributions(self.data['valid'][1], type_='validation data')

        self._keys = ['train', 'valid']

        self.kwargs = kwargs
        self.kwargs['data'] = self.data['train']
        self.kwargs['numpy_rng'] = self.numpy_rng
        self.kwargs['theano_rng'] = self.theano_rng

        self.models = []

    def __spawn_a_model(self):
        return compile_model(**copy.deepcopy(self.kwargs))

    def bootstrapped_data(self):

        data = {}
        for k, v in self.data.items():
            len_ = len(v[1])
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


    def train_(self, n_models=10, improvement_threshold=0.995,
            min_iter=2000, min_iter_increase=2, n_epochs=20):

        i = 0
        while i < n_models:
            logger.info('Optimizing Model %i' % i)
            m = self.train_a_model(
                    improvement_threshold=improvement_threshold,
                    min_iter=min_iter,
                    min_iter_increase=min_iter_increase,
                    n_epochs=n_epochs
                    )

            self.models.append(m)
            i += 1

        logger.info('done optimizing all of the models')
        return self


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
