from __future__ import print_function, division

import logging
logger = logging.getLogger(__name__)

import numpy as np
#import theano
#from theano import tensor as T
from .ensemble import Ensemble

class BoostedBagging(Ensemble):

    def __init__(self, data, sample_weights=None,
            weighting_function=None,
            training_fraction=0.8, numpy_rng=None, theano_rng=None,
            **kwargs):

        Ensemble.__init__(self, data, training_fraction=training_fraction,
                numpy_rng=numpy_rng, theano_rng=theano_rng)
        logger.info('initializing sample weights')

        if sample_weights is None:
            sample_weights = np.ones(len(data[0]))

        self._weighting_func = weighting_function

        len_ = len(self.data['train'][0])
        self._weights = dict(
                train=sample_weights[:len_]/sample_weights[:len_].sum(),
                valid=sample_weights[len_:]/sample_weights[len_:].sum()
                )

        self.kwargs = kwargs
        self.kwargs['data'] = self.data['train']
        self.kwargs['numpy_rng'] = self.numpy_rng
        self.kwargs['theano_rng'] = self.theano_rng

        self.models = []

    def update_weights(self):

        if self._weighting_func is not None:
            for k, v in self.data.items():
                Y = self.predict(v[0])
                self._weights[k] = self._weighting_func(Y, v[1])

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
            self.update_weights()
            i += 1

        logger.info('done optimizing all of the models')
        return self

