# -*- coding: utf-8 -*-
from __future__ import print_function, division

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from ..optimizers import sgd
from ..optimizers.model_optimizers import optimize_params_using_early_stopping


class BaseSupervisedModel(object):

    def __init__(self, architecture, cost_func,
                 output_type='vector', output_dtype='int32',
                 loss_func=None, confidence_func=None,
                 scoring_func=None, predict_func=None,
                 share_memory=False, early_stopping=True,
                 batch_size=50, eta_0=0.13,
                 learning_rate_adaptation=None, **lr_params):

        y = getattr(T, output_type)('y', dtype=output_dtype)

        self.arch = architecture
        self.X = self.arch.X
        self.Y = self.arch.Y
        self.params = self.arch.params

        # ------------------------------------ #
        # Define cost, loss, confidence, score #
        # ------------------------------------ #
        cost = cost_func(self.Y, y) + self.arch.reg

        if loss_func is None:
            loss = cost
        else:
            loss = loss_func(self.Y, y)

        if confidence_func is None:
            confidence_ = self.Y
        else:
            confidence_ = confidence_func(self.Y)

        if scoring_func is None:
            score_ = self.Y * y
        else:
            score_ = scoring_func(self.Y, y)

        self.score = theano.function(inputs=[self.X, y], outputs=score_)
        self.confidence = theano.function(inputs=[self.X], outputs=confidence_)

        if predict_func is None:
            pred = self.Y
        else:
            pred = predict_func(self.Y)

        self.predict = theano.function(
                        inputs=[self.X], outputs=pred
                        )

        # ------------------------------- #
        # Setting the SGD Learning Rules  #
        # ------------------------------- #
        index = T.lscalar()  # index to a batch
        grads = [T.grad(cost, p) for p in self.params]

        if learning_rate_adaptation is None:
            updates = sgd.sgd_updates(grads, self.params, eta_0=eta_0)
        elif learning_rate_adaptation == 'momentum':
            updates = sgd.sgd_updates_momentum(
                    grads, self.params, eta_0=eta_0, **lr_params
                    )
        elif learning_rate_adaptation == 'adadelta':
            updates = sgd.adadelta(
                    grads, self.params, eta_0=eta_0, **lr_params
                    )
        else:
            NotImplementedError("%s is not yet implementd" %
                                learning_rate_adaptation)

        outputs = [cost, loss]

        self.training_loss = []
        self.validation_loss = []
        self.best_validataion = None

        # ------------------------- #
        # create shared data source #
        # ------------------------- #
        self.train_X, self.train_y = [
                self.data_source(var, 'train_', borrow=share_memory)
                for var in [self.X, y]
                ]

        self.train = theano.function(
                inputs=[index],
                outputs=outputs,
                updates=updates,
                givens={
                    self.X: self.train_X[
                        index*batch_size: (index + 1) * batch_size
                        ],
                    y: self.train_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
                    }
                )

        if early_stopping:
            self.valid_X, self.valid_y = [
                self.data_source(var, 'valid_', borrow=share_memory)
                for var in [self.X, y]
                ]
            self.validate = theano.function(
                    inputs=[index],
                    outputs=loss,
                    givens={
                        self.X: self.valid_X[
                            index*batch_size: (index+1)*batch_size
                            ],
                        y: self.valid_y[
                            index * batch_size: (index + 1) * batch_size
                            ]
                        }
                    )
        self.batch_size = batch_size

    def set_params(self, values):

        for p, v in zip(self.params, values):
            p.set_value(values[v])

    def get_params(self):

        d = OrderedDict()
        for p in self.params:
            d[p.__repr__()] = p.get_value()

        return d

    def optimize_params(self, data, training_fraction=0.8,
                        improvement_threshold=0.995,
                        min_iter=2000, min_iter_increase=2, n_epochs=200):
        if training_fraction < 1.0:
            len_ = len(data[0])
            train_size = int(training_fraction * len_)
            self.train_X.set_value(data[0][: train_size])
            self.train_y.set_value(data[1][:train_size])
            self.valid_X.set_value(data[0][train_size:])
            self.valid_y.set_value(data[1][train_size:])
            tl, vl, bv = optimize_params_using_early_stopping(
                    self, improvement_threshold=improvement_threshold,
                    min_iter=min_iter, min_iter_increase=min_iter_increase,
                    n_epochs=n_epochs)
        else:
            self.train_X.set_value(data[0])
            self.train_y.set_value(data[1])

        self.training_loss += tl
        self.validation_loss += vl
        self.best_validation = bv

    def __getstate__(self):

        state = dict(self.__dict__)
        state.pop('train_X', None)
        state.pop('train_y', None)
        state.pop('valid_X', None)
        state.pop('valid_y', None)

        return state

    def __setstate__(self, d):

        self.__dict__.update(d)

    @staticmethod
    def data_source(var, prefix, borrow=False):

        return theano.shared(
                np.array(0, dtype=var.dtype, ndmin=var.ndim),
                borrow=borrow,
                name=prefix + var.name
                )