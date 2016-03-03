from __future__ import print_function, division

import theano

from ..common.random import numpy_rng_instance, theano_rng_instance

class BaseNoiseLayer(object):

    def __init__(self, X,
            numpy_rng=None, theano_rng=None, name='noise',
            deterministic=False, **kwargs):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        self.deterministic = deterministic

        self.X = X
        self.name = name


class DropoutLayer(BaseNoiseLayer):

    def __init__(self, X,
            numpy_rng=None, theano_rng=None, deterministic=False,
            dropout_proba=0.5, name='dropout'):
        BaseNoiseLayer.__init__(self, X,
                numpy_rng=numpy_rng, theano_rng=theano_rng,
                deterministic=deterministic)

        retain_proba = 1 - dropout_proba

        if (retain_proba == 1) or self.deterministic:
            self.Y = self.X
        else:
            self.Y = self.X * self.theano_rng.binomial(
                self.X.shape, p=retain_proba,
                dtype=theano.config.floatX
                )
