# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import theano

from ..common import functions as uf
from ..common.random import numpy_rng_instance, theano_rng_instance


class BaseGLMLayer(object):

    def __init__(self, X, input_shape, output_shape, numpy_rng=None,
                 theano_rng=None, W=None, b=None, name='layer',
                 activation=None, l1_reg=None, l2_reg=None):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        self.input_shape = input_shape
        self.output_shape = output_shape

        input_shape = self.input_shape[0]
        output_shape = self.output_shape[0]

        self.W = initial_W(
            numpy_rng,
            W_shape=(input_shape, output_shape), W=W,
            activation=activation, name=name
            )

        self.b = initial_b(
            b_shape=(output_shape, ),
            b=b, name=name
            )

        self.X = X

        if activation is None:
            self.Y = uf.linear(self.X, self.W, self.b)
        else:
            self.Y = activation(uf.linear(self.X, self.W, self.b))

        self.reg = 0.
        if l1_reg is not None:
            self.reg += l1_reg * abs(self.W).sum()
        if l2_reg is not None:
            self.reg += l2_reg * (self.W ** 2).sum()

        self.params = [self.W, self.b]

    def __repr__(self):
        return self.__class__.__name__


class SoftplusLayer(BaseGLMLayer):

    def __init__(self, X, input_shape, output_shape, numpy_rng, theano_rng,
                 W=None, b=None, name='layer'):
        BaseGLMLayer.__init__(
                self, X, input_shape, output_shape, numpy_rng, theano_rng,
                W=W, b=b, name=name, activation=uf.softplus
                )


class LinearLayer(BaseGLMLayer):

    def __init__(self, X, input_shape, output_shape, numpy_rng, theano_rng,
                 W=None, b=None, name='layer', l1_reg=None, l2_reg=None):
        BaseGLMLayer.__init__(
                self, X, input_shape, output_shape, numpy_rng, theano_rng,
                W=W, b=b, name=name, l1_reg=l1_reg, l2_reg=l2_reg,
                activation=None
                )


class TanhLayer(BaseGLMLayer):

    def __init__(self, X, input_shape, output_shape, numpy_rng, theano_rng,
                 W=None, b=None, name='layer', l1_reg=None, l2_reg=None):
        BaseGLMLayer.__init__(
                self, X, input_shape, output_shape, numpy_rng, theano_rng,
                W=W, b=b, name=name, l1_reg=l1_reg, l2_reg=l2_reg,
                activation=uf.tanh
                )


class SigmoidLayer(BaseGLMLayer):

    def __init__(self, X, input_shape, output_shape, numpy_rng, theano_rng,
                 W=None, b=None, name='layer', l1_reg=None, l2_reg=None):
        BaseGLMLayer.__init__(
                self, X, input_shape, output_shape, numpy_rng, theano_rng,
                W=W, b=b, name=name, l1_reg=l1_reg, l2_reg=l2_reg,
                activation=uf.sigmoid
                )


class SoftmaxLayer(BaseGLMLayer):

    def __init__(self, X, input_shape, output_shape, numpy_rng, theano_rng,
                 W=None, b=None, name='layer'):
        BaseGLMLayer.__init__(
                self, X, input_shape, output_shape, numpy_rng, theano_rng,
                W=W, b=b, name=name, activation=uf.softmax
                )


def initial_W(numpy_rng, W_shape=None, W=None, activation=None,
              name='layer', borrow=False):

    if (W_shape is None) and (W is None):
        raise ValueError(
            "Either `W_shape` or `W` has to be specified"
        )

    if W is None:
        if activation in (uf.softmax, uf.softplus, None):
            W = np.zeros(W_shape, dtype=theano.config.floatX)
        else:
            W = np.asarray(
                numpy_rng.uniform(
                    low=-np.sqrt(6. / np.sum(W_shape)),
                    high=np.sqrt(6. / np.sum(W_shape)),
                    size=W_shape
                ),
                dtype=theano.config.floatX
            )
            if activation == uf.sigmoid:
                W *= 4
            elif activation == uf.tanh:
                pass
            else:
                raise NotImplementedError(
                    "Activation: %s is not implemented." % repr(activation)
                )
    name_ = 'W_{}'.format(name)
    return theano.shared(value=W, name=name_, borrow=borrow)


def initial_b(b_shape=None, b=None, name='layer', borrow=False):

    if (b_shape is None) and (b is None):
        raise ValueError(
            "Either `b_shape` or `b` has to be specified"
        )

    if b is None:
        b = np.zeros(b_shape, dtype=theano.config.floatX)

    name_ = 'b_{}'.format(name)
    return theano.shared(value=b, name=name_, borrow=borrow)
