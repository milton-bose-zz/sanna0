# -*- coding: utf-8 -*-
from __future__ import print_function, division

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from ..optimizers import sgd
from ..utils.data_processing import split_dataset
from ..optimizers.model_optimizers import optimize_params as optimize


class BaseSupervisedModel(object):

    def __init__(self, architecture, cost_func,
                 output_type='vector', output_dtype='int32',
                 loss_func=None, confidence_func=None,
                 scoring_func=None, predict_func=None,
                 share_memory=False, early_stopping=True,
                 batch_size=50, eta_0=0.13,
                 learning_rate_adaptation=None, **kwargs):

        y = getattr(T, output_type)('y', dtype=output_dtype)

        self.early_stopping = early_stopping
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

        self.confidence = theano.function(
                inputs=[self.X], outputs=confidence_
                )

        if scoring_func is None:
            try:
                score_ = self.Y[T.arange(self.Y.shape[0]), y]
            except TypeError:
                score_ = cost_func(self.Y, y)
        else:
            score_ = scoring_func(self.Y, y)

        self.score = theano.function(inputs=[self.X, y], outputs=score_)


        if predict_func is None:
            pred = self.Y
        else:
            pred = predict_func(self.Y)
        self.arch.freeze_layers()
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
                    grads, self.params, eta_0=eta_0, **kwargs
                    )
        elif learning_rate_adaptation == 'adadelta':
            updates = sgd.adadelta(
                    grads, self.params, eta_0=eta_0, **kwargs
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
        self.arch.unfreeze_layers()
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

    def train_(self, data, training_fraction=0.8,
            improvement_threshold=0.995,
            min_iter=2000, min_iter_increase=2, n_epochs=200):

        train_data = split_dataset(
                data, training_fraction=training_fraction,
                numpy_rng=self.arch.numpy_rng)

        self.optimize_params(train_data,
                improvement_threshold=improvement_threshold,
                min_iter=min_iter,
                min_iter_increase=min_iter_increase,
                n_epochs=n_epochs
                )


    def optimize_params(self, data, improvement_threshold=0.995,
                        min_iter=2000, min_iter_increase=2, n_epochs=200):

        self.load_datasets(data)

        tl, vl, bv = optimize(
                self, improvement_threshold=improvement_threshold,
                min_iter=min_iter, min_iter_increase=min_iter_increase,
                n_epochs=n_epochs,
                early_stopping=self.early_stopping)
        self.training_loss += tl
        self.validation_loss += vl
        self.best_validation = bv

    def __getstate__(self):

        self.train_X.set_value(
                np.array(0, dtype=self.train_X.dtype, ndmin=self.train_X.ndim)
                )
        self.train_y.set_value(
                np.array(0, dtype=self.train_y.dtype, ndmin=self.train_y.ndim)
                )
        if self.early_stopping:
            self.valid_X.set_value(
                np.array(0, dtype=self.valid_X.dtype, ndmin=self.valid_X.ndim)
                )
            self.valid_y.set_value(
                np.array(0, dtype=self.valid_y.dtype, ndmin=self.valid_y.ndim)
                )
        state = dict(self.__dict__)
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

    def load_datasets(self, data):

        self.train_X.set_value(data['train'][0])
        self.train_y.set_value(data['train'][1])
        if self.early_stopping:
            self.valid_X.set_value(data['valid'][0])
            self.valid_y.set_value(data['valid'][1])

