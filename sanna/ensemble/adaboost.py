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

        self.data = split_dataset(data, train_fraction=training_fraction,
                numpy_rng=numpy_rng)

        self.class_weight=class_weight

        self._keys = ['train', 'valid']
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

        def spawn_a_model(model_kwargs):
            return compile_model(**model_kwargs)
        self.__model = spawn_a_model

        self.models = []

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
        logger.info('compling... ')
        model = self.__model(copy.deepcopy(self.kwargs))
        logger.info('... done!')
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

        #compute score
        score = model.confidence(data[0])
        y = binarize(data[1], score.shape[1])
        score = score * (2 * y - 1)
        #print(score)

        eps = (D * (1 - score.sum(axis=1))).sum() / 2
        beta = eps / (1 - eps) # beta < 1.0

        scale = beta ** ((1 + score) / 2)

        w *= scale # inpace operations
        w /= w.sum() # inplace operations

        return eps, beta

    def train(self, n_models=2, improvement_threshold=0.995,
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
            for k in self.keys:
                eps, beta = self.update_weights(self, m, k)
                logger.info('{0} ==> epsilon: {1}, beta: {2}'.format(
                    k, eps, beta
                    )
                    )
                self.eps[k].append(eps)
                self.neg_log_beta[k].append(- np.log(beta))

            self.models.append(m)

        logger.info('done optimizing all of the models')
        return self


    def confidence(self, X, key='train'):

        Y = np.zeros((len(X), 1))
        for m, nlb in zip(self.models, self.neg_log_beta[key]):
            Y = Y + m.confidence(X) * nlb

        return Y

    def predict(self, X, key='train'):

        Y = self.confidence(X, key)
        return Y.argmax(axis=1)
