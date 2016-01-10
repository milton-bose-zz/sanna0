# -*- coding: utf-8 -*-
from __future__ import print_function, division

# import numpy as np
import theano

from .base_arch import NetworkArchitecture
from .. import layers


class FeedForwardNetwork(NetworkArchitecture):

    def __init__(self, X, network_description,
                 numpy_rng=None, theano_rng=None):

        NetworkArchitecture.__init__(
                self, X, numpy_rng=numpy_rng, theano_rng=theano_rng
                )

        for layer_descr in network_description:
            lyr_cls = layer_descr.pop('layer_class')
            if isinstance(lyr_cls, str):
                lyr_cls = getattr(layers, lyr_cls)
            if len(self.layers) == 0:
                input_ = X
            else:
                input_ = self.layers[-1].Y
            self.layers.append(
                    lyr_cls(
                        input_, numpy_rng=self.numpy_rng,
                        theano_rng=self.theano_rng, **layer_descr
                        )
                    )
            self.reg += self.layers[-1].reg
            self.params += self.layers[-1].params

        self.Y = self.layers[-1].Y

    def __repr__(self):
        return theano.pprint(self.Y)
