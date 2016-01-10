# -*- coding: utf-8 -*-
from __future__ import print_function, division

from collections import OrderedDict

from ..common.random import numpy_rng_instance, theano_rng_instance


class NetworkArchitecture(object):

    def __init__(self, X, numpy_rng=None, theano_rng=None):

        self.numpy_rng = numpy_rng_instance(numpy_rng)
        self.theano_rng = theano_rng_instance(theano_rng)

        self.layers = []
        self.params = []
        self.reg = 0.

        self.X = X
        self.Y = None

    def set_params(self, values):

        for p, v in zip(self.params, values):
            p.set_value(values[v])

    def get_params(self):

        d = OrderedDict()
        for p in self.params:
            d[p.__repr__()] = p.get_value()

        return d
