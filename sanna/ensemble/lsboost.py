from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
#import copy
from .ensemble import Ensemble
#from ..utils.model_compilers import compile_model

class LSBoost(Ensemble):

    def __init__(self, data, training_fraction=0.8, numpy_rng=None,
            theano_rng=None, **kwargs):

        Ensemble.__init__(self, data, training_fraction=training_fraction,
                numpy_rng=numpy_rng, theano_rng=theano_rng)

        self.kwargs = kwargs
        self.kwargs['data'] = self.data['train']
        self.kwargs['numpy_rng'] = self.numpy_rng
        self.kwargs['theano_rng'] = self.theano_rng

        self.models = []

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

    def bootstrapped_data(self):

        data = {}
        for k, v in self.data.items():
            data[k] = (v[0], v[1] - self.predict(v[0]))

        return data

    def confidence(self, X):

        Y = np.zeros((len(X), 1))
        for m in self.models:
            Y = Y + m.confidence(X)
        return Y

    def predict(self, X):

        return self.confidence(X)
