#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import logging
import argparse
import yaml

from .helpers import misc as hlp
from .helpers import loaders
from .helpers import model_compilers as compiler
from .common.eval_metrics import confusion_matrix

def run():
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser(
            description="Train a simple feed-forward neural network."
            )
    parser.add_argument(
            "model", type=str,
            help=("relative path to pre-trained model or model configuration"
                " YAML file")
            )
    parser.add_argument(
            "-o", "--optimize", action='store_true',
            help="bool",
            )
    parser.add_argument(
            '-l', '--log', type=str, default=None,
            help='path to log file'
            )
    parser.add_argument(
            '-e', '--evaluation', type=str, default=None,
            help='YAML file'
            )

    args = parser.parse_args()

    hlp.setup_basic_logging(log_file=args.log)

    if args.model[-4:] == '.pkl':
        model = loaders.read_file(args.model, file_type='pkl')
    else:
        cfg = yaml.load(loaders.read_file(args.model, file_type='txt'))
        data = cfg['data']
        data = loaders.load_datasets(
                data['filepath'], data.get('processor', None)
                )
        seeds = cfg.get('rng_seeds', None)
        if seeds is not None:
            numpy_rng = seeds['numpy']
            theano_rng = seeds['theano']

        model = compiler.compile_model(
                cfg['architecture'], data, cost_func=cfg['cost_function'],
                loss_func=cfg.get('loss_function', None),
                confidence_func=cfg.get('confidence_function', None),
                predict_func=cfg.get('predict_function', None),
                scoring_func=cfg.get('scoring_func', None),
                model_class=cfg['class'],
                numpy_rng=numpy_rng, theano_rng=theano_rng,
                gd_params=cfg.get('gradient_descent', None)
                )

        if args.optimize:
            logging.info('Optimizing The model')
            model.optimize_params(data['train'], **cfg['optimization_params'])


def evaluation(data, model, eval_metrics=[], confusion=False,
               confusion_kwrgs={}):

    pred = model.predict(data[0])
    eval_ = {}
    for m in eval_metrics:
        k = m['name']
        f = hlp.construct_eval_metrics(m)
        eval_[k] = f(pred, data[1])

    if confusion:
        cm = confusion_matrix(data[1], pred)
        cms = confusion_matrix(data[1], pred, scaled=True)
    else:
        cm = None
        cms = None
    return eval_, cm, cms
