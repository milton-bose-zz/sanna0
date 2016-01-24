# -*- coding: utf-8 -*-
from __future__ import print_function, division

import yaml
import logging
import inspect
import copy

from .helpers import loaders
from .helpers import misc as hlp
from .utils import model_compilers as compiler
from .common.random import (numpy_rng_instance, theano_rng_instance)
from . import ensemble


logger = logging.getLogger(__name__)

class Sankara(object):

    def __init__(self, cfg_yaml, logging_stream=None):

        self.logs = ''

        self.cfg_yaml = cfg_yaml
        self.cfg = yaml.load(cfg_yaml)

        data = self.cfg['data']
        self.data = loaders.load_datasets(
                data['filepath'], data.get('processor', None)
                )

        logger.info('seeding random number generators')
        seeds = self.cfg.get('rng_seeds', None)
        numpy_rng = numpy_rng_instance(seeds['numpy'])
        theano_rng = theano_rng_instance(seeds['theano'])

        model_kwargs = dict(
                data=self.data['train'],
                arch_descr=self.cfg['architecture'],
                cost_func=self.cfg['cost_function'],
                loss_func=self.cfg.get(
                    'loss_function', None
                    ),
                confidence_func=self.cfg.get(
                        'confidence_function', None
                        ),
                predict_func=self.cfg.get('predict_function', None),
                scoring_func=self.cfg.get('scoring_function', None),
                model_class=self.cfg['class'],
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                gd_params=self.cfg.get('gradient_descent', None)
                )

        ensemble_ = self.cfg.get('ensemble', None)
        if ensemble_ is None:
            self.optimize_kwargs = self.cfg.pop('optimization_params', {})
            self.optimize_kwargs['data'] = self.data['train']
            self.model = compiler.compile_model(**model_kwargs)
        else:
            self.optimize_kwargs = self.cfg.pop('optimization_params', {})
            model_kwargs['training_fraction'] = self.optimize_kwargs.pop(
                    'training_fraction', 0.8
                    )

            model = getattr(ensemble, ensemble_['method'])
            self.model = model(**model_kwargs)

        if logging_stream is not None:
            self.logs += logging_stream.getvalue()
            self.logs += '\n'
            logging_stream.truncate(0)
            logging_stream.seek(0)
        #print(self.logs)

    def train(self, logging_stream=None, **kwargs):

        args = inspect.getargspec(self.model.train_).args
        for k in kwargs.keys():
            if k not in args:
                kwargs.pop(k)
        optimize_kwargs = copy.deepcopy(self.optimize_kwargs)
        optimize_kwargs.update(kwargs)
        logger.info('current training params: {0}'.format(
            optimize_kwargs
            )
            )
        self.model.train_(**optimize_kwargs)

        if logging_stream is not None:
            self.logs += logging_stream.getvalue()
            self.logs += '\n'
            logging_stream.truncate(0)
            logging_stream.seek(0)

        return self.model

    def evaluate(self, eval_yaml, ensemble_only=True, logging_stream=None,
            data=None):

        cfg = yaml.loag(eval_yaml)

        if data is None:
            data = self.data['eval']

        if ensemble_only:
            eval_ = hlp.evaluation(
                    data, self.model, eval_metrics=cfg.get('metrics', []),
                    confusion=cfg.get('confusion_matrix', False)
                    )
        else:
            eval_ = [
                    hlp.evaluation(
                    data, m, eval_metrics=cfg.get('metrics', []),
                    confusion=cfg.get('confusion_matrix', False)
                    ) for m in (self.model.models[:] + [self.model])
                    ]

        return eval_




    def predict(self, X, logging_stream=None, **kwargs):
        pass


