# -*- coding: utf-8 -*-
from __future__ import print_function, division

import yaml
import logging; logger = logging.getLogger(__name__)

from .helpers import loaders
from .utils import model_compilers as compiler
from .common.random import (numpy_rng_instance, theano_rng_instance)
from . import ensemble




class Sankara(object):

    def __init__(self, cfg_yaml):

        self.logs = ''

        self.cfg_yaml = cfg_yaml
        self.cfg = yaml.load(cfg_yaml)

        data = self.cfg['data']
        self.data = loaders.load_datasets(
                data['filepath'], data.get('processor', None)
                )

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
            model = compiler.compile_model(**model_kwargs)
        else:
            self.optimize_kwargs = self.cfg.pop('optimization_params', {})
            model_kwargs['training_fraction'] = self.optimize_kwargs.pop(
                    'training_fraction', 0.8
                    )

            model = getattr(ensemble, ensemble_['method'])
            model = model(**model_kwargs)

    def train(self, logging_stream=None, **kwargs):
        pass

    def evaluate(self, logging_stream=None, **kwargs):
        pass

    def predict(self, X, logging_stream=None, **kwargs):
        pass



