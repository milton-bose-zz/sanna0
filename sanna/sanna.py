#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import logging
import argparse
import pickle
import yaml
import theano
import theano.tensor as T

from .arch.feed_forward import FeedForwardNetwork
from . import models
from ._helper import sanna_helper as hlp
from .common.eval_metrics import confusion_matrix

bar = '#' * 80
config_bar = '#' * 33 + ' CONFIGURATION ' + '#' * 33


def show_header(model_name, config_text=None):

    if config_text is None:
        logging.info('running model: {}'.format(model_name))
    else:
        logging.info(
                'running model: {0}\n{1}\n{2}\n{3}'.format(
                    model_name, config_bar, config_text.strip(),
                    bar
                    )
                )


def run():
    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser(
            description="Train a simple feed-forward neural network."
            )

    parser.add_argument(
            "-c", "--config", type=str, default=None,
            help="YAML file with the model configuration."
            )
    parser.add_argument(
            "-m", "--model", type=str, default=None,
            help="relative path to pre-trained model")
    parser.add_argument(
            "-o", "--optimize", action='store_true',
            help="bool",
            )
    parser.add_argument(
            '-l', '--log', type=str, default=None,
            help='path to log file'
            )

    args = parser.parse_args()
    log_file = args.log
    if args.config is not None:
        config_text = hlp.read_file(args.config)
        config = yaml.load(config_text)
        log_file = config['log_filename']
    else:
        if args.model is None:
            raise SyntaxError(
                'Either model config or a pickled model must be given'
                )

    optimize = args.optimize
    pickled_model = args.model
    if pickled_model is not None:
        model_name = pickled_model
    else:
        model_name = config['model_name']

    hlp.setup_basic_logging(log_file=log_file)
    show_header(model_name, config_text)

    logging.info('loading datasets')
    data = hlp.load_datasets(config['datasets'])

    if pickled_model is None:
        model = compile_model(config, data)
        optimize = True
    else:
        with open(pickled_model, 'rb') as f:
            model = pickle.load(f)

    if optimize:
        logging.info('Optimizing the model parameters')
        model.optimize_params(data['training'],
                **config['optimization_params']
                )

    eval_config = config['evaluation']
    confusion_mat = eval_config.get('confusion_matrix', False)
    eval_metrics = eval_config.get('metrics', [])

    msg = "\n\nPerformance Evaluation\n======================\n\n"

    if 'eval' in data.keys():
        eval_, cm, cms = evaluation(
                data['eval'], model, eval_metrics, confusion_mat
                )
        msg += "Test Data (%i instances)\n\n" % len(data['eval'][1])
        for k, v in eval_.items():
            msg += '{0}: {1}\n'.format(k, v)
        msg += '\n'
        if cm is not None:
            msg += 'Confusion Matrix: (actuals along row)\n'
            msg += '-------------------------------------\n'
            msg += hlp.pandas_repr(cm, display__width=200, precision=4)
            msg += '\n\n'
            msg += 'Confusion Matrix Normalized Along Row:\n'
            msg += '--------------------------------------\n'
            msg += hlp.pandas_repr(cms, display__width=200, precision=4)
            msg += '\n\n'
        else:
            msg += 'No Evaluation data provided\n\n'

    logging.info(msg)

    if config['pickle']:
        logging.info(
                'pickling the trained model to file: %s.pkl' % (
                    config['model_name']
                    )
                )
        with open(config['model_name'] + '.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def compile_model(config, data):

    logging.info('setting up the random number generators')
    numpy_rng, theano_rng = hlp.initialize_rngs(
            config['rng_seeds']['numpy'], config['rng_seeds']['theano']
            )

    logging.info('Constructing the architecture')
    architecture = config['model']['architecture']
    input_type = hlp.get_tensor_type(data['training'][0])
    X = getattr(T, input_type)('X', dtype=theano.config.floatX)
    arch = FeedForwardNetwork(
            X, architecture, numpy_rng=numpy_rng, theano_rng=theano_rng
            )

    logging.info('Compiling the model')
    output_type = hlp.get_tensor_type(data['training'][1])
    output_dtype = str(data['training'][1].dtype)
    ModelClass = getattr(models, config['model']['class'])
    cost_func = hlp.construct_function(config['model']['cost_function'])
    confidence_func = hlp.construct_function(
            config['model']['confidence_function']
            )
    loss_func = hlp.construct_function(config['model']['loss_function'])
    scoring_func = hlp.construct_function(
            config['model']['scoring_function']
            )
    predict_func = hlp.construct_function(
            config['model']['predict_function']
            )
    model_params = dict(
        cost_func=cost_func,
        loss_func=loss_func,
        predict_func=predict_func,
        scoring_func=scoring_func,
        confidence_func=confidence_func,
        output_type=output_type,
        output_dtype=output_dtype
        )
    gd_params = config['model']['gradient_descent']
    model_params.update(gd_params)
    model = ModelClass(arch, **model_params)

    return model


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
