from __future__ import print_function, division

from ..common.random import numpy_rng_instance, theano_rng_instance

class BaseNoiseLayer(object):

    def __init__(self, X, input_shape, output_shape,
            numpy_rng=None, theano_rng=None,
            deterministic=False, **kwargs):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.deterministic = deterministic


class DropoutLayer(BaseNoiseLayer):

    def __init__(self, X, input_shape, output_shape,
            numpy_rng=None, theano_rng=None, deterministic=False, p=0.5):
        BaseNoiseLayer.__init__(self, X, input_shape, output_shape,
                numpy_rng=numpy_rng, theano_rng=theano_rng,
                deterministic=deterministic)

        retain_proba = 1 - p
