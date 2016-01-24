#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import io
import os
from os.path import basename
import sys
import click
import logging
import time

from .sankara import Sankara
from .helpers import misc as hlp, loaders


logging.Formatter.converter = time.gmtime
sys.path.append(os.getcwd())

__MARGIN = 2
__spc = ' ' * __MARGIN
__LOG = io.StringIO()
__HNDL = logging.StreamHandler(__LOG)


@click.group(invoke_without_command=True, chain=True)
@click.argument('model_config', type=click.Path())
@click.option('--log', type=click.Path(), help='log filepath')
@click.option('--pickle/--no-pickle', default=True,
        help='whether to pickle or no.')
@click.pass_context
def karma(ctx, model_config, log=None, pickle=True):
    """build the model according to MODEL_CONFIG recepie
    """
    ctx.obj = dict(
            pickle=pickle
            )

    if log is None:
        handlers = [__HNDL, logging.StreamHandler(sys.stdout)]
    else:
        handlers = [__HNDL, logging.FileHandler(log)]
    hlp.setup_basic_logging(handlers=handlers)

    model_name = os.path.splitext(basename(model_config))[0]

    if model_config.endswith('.yaml') or model_config.endswith('.yml'):
        cfg_yaml = loaders.read_file(model_config, file_type='txt')

        logging.info(banner(model_name))
        snkr = Sankara(cfg_yaml, logging_stream=__LOG)
    else:
        logging.info(banner(model_name))
        logging.info('unpickling the model ... ')
        snkr = loaders.read_file(model_config, file_type='pkl')
        logging.info('... done!')

    ctx.obj['snkr'] = snkr
    ctx.obj['model_name'] = model_name

@karma.command(short_help='train the model')
@click.option('--n_models', type=int, default=1,
        help='number of models')
@click.option('--min_iter', type=int, default=2000,
        help='number of iteration')
@click.option('--n_epochs', type=int, default=10,
        help='number of epochs')
@click.pass_context
def train(ctx, n_models, min_iter, n_epochs):
    snkr = ctx.obj['snkr']
    pickle = ctx.obj['pickle']
    snkr.train(logging_stream=__LOG, n_models=n_models,
            min_iter=min_iter, n_epochs=n_epochs)

    if pickle:
        loaders.pickle_file(snkr, ctx.obj['model_name']+'.pkl')



def banner(model_name):

    banner = (
            '\n' +
            __spc + '+' + '-' * (4 + len(model_name)) + '+\n' +
            __spc + '|' + ' ' * 2 + model_name + ' ' *2 + '|\n' +
            __spc + '+' + '-' * (4 + len(model_name)) + '+\n'
            )
    return banner


