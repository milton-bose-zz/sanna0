from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
#import theano
#from theano import tensor as T
import copy

from .utils import initialize_class_weights
from ..common.random import numpy_rng_instance, theano_rng_instance
from ..utils.data_processing import (randomized_data_index,
                                    split_dataset, binarize)
from ..utils.model_compilers import compile_model

class AdaBoostM2(object):

    def __init__(self, data, class_weight=None,
            training_fraction=0.8, numpy_rng=None, theano_rng=None,
            **kwargs):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        logger.info('splitting the data into `train` and `valid` sets')
        self.data = split_dataset(data, training_fraction=training_fraction,
                numpy_rng=self.numpy_rng)

        log_distributions(self.data['train'][1], type_='training data')
        log_distributions(self.data['valid'][1], type_='validation data')

        self._keys = ['train', 'valid']

        logger.info('initializing sample weights')
        self.class_weight=class_weight
        self._weights = {}
        self.eps = {}
        self.neg_log_beta = {}
        for k in self._keys:
            self._weights[k] = initialize_class_weights(
                    self.data[k][1], class_weight=class_weight,
                    return_numpy_object=True
                    )
            self.eps[k] = []
            self.neg_log_beta[k] = []

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
            p = self._weights[k].sum(axis=-1)
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

    def update_weights(self, model, key):

        data = self.data[key]
        w = self._weights[key]
        D = w.sum(axis=1)/w.sum()
        q = w/w.sum(axis=1, keepdims=True)
        q[np.arange(len(q)), data[1]] = 1.0
        #print('q: ', q)
        #print(D)

        #compute score
        score = model.confidence(data[0])
        y = binarize(data[1], score.shape[1])
        score = score * (2 * y - 1)
        #print(score)

        eps = (D * (1 - np.sum(score * q, axis=1))).sum() / 2
        beta = eps / (1 - eps) # beta < 1.0

        scale = beta ** ((1 + score) / 2)

        w *= scale # inpace operations
        w /= w.sum() # inplace operations

        return eps, beta

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


    def confidence(self, X, key='valid'):

        Y = np.zeros((len(X), 1))
        for m, nlb in zip(self.models, self.neg_log_beta[key]):
            Y = Y + m.confidence(X) * nlb

        return Y

    def predict(self, X, key='valid'):

        Y = self.confidence(X, key)
        return Y.argmax(axis=1)


def log_distributions(data, type_='training sample'):

    count = np.unique(data, return_counts=True)[1]
    logger.info('{0} distribution: {1}'.format(type_, count))
