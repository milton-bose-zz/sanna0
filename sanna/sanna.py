#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import logging
import argparse
import yaml
import pickle

from .helpers import misc as hlp
from .helpers import loaders
from .utils import model_compilers as compiler
from .common.random import (numpy_rng_instance, theano_rng_instance)
from .utils.data_processing import split_dataset


def run():

    space = ' ' * 2
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
            '-l', '--log', type=str, default=None,
            help='path to log file'
            )
    parser.add_argument(
            '-e', '--evaluation', type=str, default=None,
            help='YAML file'
            )

    args = parser.parse_args()

    hlp.setup_basic_logging(log_file=args.log)

    model_path = args.model
    model_name = model_path.split('/')[-1]
    model_name = model_name.split('.')[0]

    logging.info(
            space + "\n"+
            space + "# " + "="* len(model_name) + " #\n"+
            space + "# " + model_name + " #\n"+
            space + "# " + "="* len(model_name) + " #\n"
            )

    if model_path[-4:] == '.pkl':
        model = loaders.read_file(model_path, file_type='pkl')
        pickled = False
        data = None
    else:
        cfg = yaml.load(loaders.read_file(model_path, file_type='txt'))
        data = cfg['data']
        data = loaders.load_datasets(
                data['filepath'], data.get('processor', None)
                )

        seeds = cfg.get('rng_seeds', None)
        logging.info('setting up the random number generators')
        numpy_rng = numpy_rng_instance(seeds['numpy'])
        theano_rng = theano_rng_instance(seeds['theano'])

        model = compiler.compile_model(
                cfg['architecture'], data['train'],
                cost_func=cfg['cost_function'],
                loss_func=cfg.get('loss_function', None),
                confidence_func=cfg.get('confidence_function', None),
                predict_func=cfg.get('predict_function', None),
                scoring_func=cfg.get('scoring_func', None),
                model_class=cfg['class'],
                numpy_rng=numpy_rng, theano_rng=theano_rng,
                gd_params=cfg.get('gradient_descent', None)
                )
        pickled = True

        logging.info('Optimizing The model')
        oparams = cfg.get('optimization_params', {})
        train_fraction = oparams.pop(oparams, 0.8)
        train_data = split_dataset(data['train'],
                train_fraction=train_fraction, numpy_rng=numpy_rng)

        model.optimize_params(train_data, **oparams)

    if pickled:
        logging.info('Pickling the model at {}'.format(model_name + '.pkl'))
        with open(model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


    if args.evaluation is not None:
        eval_cfg = yaml.load(
                loaders.read_file(args.evaluation, file_type='txt')
                )
        if data is None:
            data = eval_cfg['data']
            data = loaders.load_datasets(
                    data['filepath'], data.get('processor', None)
                    )
            data = data['eval']
        else:
            data = data['eval']

        eval_ = hlp.evaluation(
                data, model, eval_metrics=eval_cfg.get('metrics', []),
                confusion=eval_cfg.get('confusion_matrix', False)
                )

        msg = "\n"
        msg += space + '+------------+\n'
        msg += space + '| EVALUATION |\n'
        msg += space + '+------------+\n'
        for k, v in eval_['eval_metrics'].items():
            msg += space + k +': {}\n'.format(v)

        if eval_['cm'] is not None:
            msg += '\n'
            msg += space + '+--------------------------------------+\n'
            msg += space + '| Confusion Matrix (Actuals along row) |\n'
            msg += space + '+--------------------------------------+\n'
            msg += hlp.pandas_repr(eval_['cm'], display__width=200,
                    precision=4, display__colheader_justify='right')
            msg += '\n\n' + space + 'Scaled Along row:\n'
            msg += space + '-----------------\n'
            msg += hlp.pandas_repr(eval_['cms'], display__width=200,
                    precision=4,
                    display__colheader_justify='right')
            msg += '\n\n'
            msg += '  ' + 'xo' * 39 + '\n'

        logging.info(msg)

