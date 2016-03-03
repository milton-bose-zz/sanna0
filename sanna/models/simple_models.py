from __future__ import division, print_function

import theano
import numpy as np
import theano.tensor as T
from theano import shared

class MinEuclideanDistance(object):

    def __init__(self, Y, y, init_param=1.0, p=2):
        self.param = shared(
                value=np.array(init_param, dtype=theano.config.floatX),
                name='rho'
                )
        y = T.constant(y)
        Y = T.constant(Y)
        f = T.mean((y - self.param * Y) ** p)

        grad = T.grad(f, self.param)
        hess = T.grad(grad, self.param)

        updates =[
                (self.param, self.param - (p - 1) * grad / hess)
                ]

        self.min_ = theano.function(
                inputs=[],
                outputs=grad,
                updates=updates)


    def update_param(self):

        grad = self.min_()

        return grad



    def get_param(self):

        return self.param.get_value()







