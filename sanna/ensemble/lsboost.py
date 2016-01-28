from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
import copy

from ..common.random import numpy_rng_instance, theano_rng_instance
from ..utils.data_processing import (split_dataset)
from ..utils.model_compilers import compile_model



class LSBoost(object):

    def __init__(self, data, training_fraction=0.8, numpy_rng=None,
            theano_rng=None, **kwargs):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        logger.info('splitting the data into `train` and `valid` sets')
        self.data = split_dataset(data, training_fraction=training_fraction,
                numpy_rng=numpy_rng)

        self._keys = ['train', 'valid']

        logger.info('initializing sample weights')


        self.kwargs = kwargs
        self.kwargs['data'] = self.data['train']
        self.kwargs['numpy_rng'] = self.numpy_rng
        self.kwargs['theano_rng'] = self.theano_rng

        self.models = []

    def __spawn_a_model(self):

        return compile_model(**copy.deepcopy(self.kwargs))

    def train_a_model(self, improvement_threshold=0.995,
            min_iter=2000, min_iter_increase=2, n_epochs=20):

        model = self.__spawn_a_model()
        model.optimize_params(
                self.data,
                improvement_threshold=improvement_threshold,
                min_iter=min_iter,
                min_iter_increase=min_iter_increase,
                n_epochs=n_epochs
                )
        return model


    def train_(self, n_models=2, improvement_threshold=0.995,
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
            for k in self._keys:
                eps, beta = self.update_weights(m, k)
                logger.info('{0} ==> epsilon: {1}, beta: {2}'.format(
                    k, eps, beta
                    )
                    )
                self.eps[k].append(eps)
                self.neg_log_beta[k].append(- np.log(beta))

            self.models.append(m)
            i += 1

        logger.info('done optimizing all of the models')
        return self









