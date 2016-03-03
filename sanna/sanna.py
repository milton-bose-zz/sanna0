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

sys.setrecursionlimit(50000)

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
@click.option('--n_epochs', type=int, default=20,
        help='number of epochs')
@click.option('--supplied/--use-cfg', default=False)
@click.pass_context
def train(ctx, n_models, min_iter, n_epochs, supplied):
    snkr = ctx.obj['snkr']
    pickle = ctx.obj['pickle']
    if supplied:
        snkr.train(logging_stream=__LOG, n_models=n_models,
            min_iter=min_iter, n_epochs=n_epochs)
    else:
        snkr.train(logging_stream=__LOG, n_models=n_models)

    if pickle:
        logging.info('pickling the model to %s' % (
            ctx.obj['model_name'] + '.pkl'
            )

            )
        loaders.pickle_file(snkr, ctx.obj['model_name']+'.pkl')


@karma.command(short_help='evaluation')
@click.argument('eval_cfg', type=click.Path())
@click.option('--data', type=click.Path())
@click.pass_context
def evaluate(ctx, eval_cfg, data=None):

    snkr = ctx.obj['snkr']

    eval_yaml = loaders.read_file(eval_cfg, 'txt')
    if data is None:
        #print(snkr.data)
        data = snkr.data['eval']
    else:
        data = loaders.read_file(data, 'pkl')
        try:
            data = data['eval']
        except:
            pass

    eval_ = snkr.evaluate(eval_yaml, logging_stream=__LOG, data=data)

    eval_msg = "\n"
    eval_msg += __spc + '+------------+\n'
    eval_msg += __spc + '| EVALUATION |\n'
    eval_msg += __spc + '+------------+\n'
    eval_msg += __spc + 'number of samples: ' + str(len(data[0])) + '\n'
    eval_msg += '\n'

    if isinstance(eval_, list):
        last = eval_.pop()
        i = 0
        for e in eval_:
            title = 'MODEL %i' % i
            msg = evaluation_printout(e, title)
            eval_msg += msg
            i += 1
        title = 'MODEL ENSEMBLE'
    else:
        last = eval_
        title = None

    msg = evaluation_printout(last, title=title)
    eval_msg += msg
    logging.info(eval_msg)


def evaluation_printout(eval_, title=None):

    if title is None:
        msg = '\n'
    else:
        msg = '\n'
        msg += __spc + title  + '\n'
        msg += __spc + '=' * len(title) + '\n'

    for k, v in eval_['eval_metrics'].items():
        msg += __spc + k +': {}\n'.format(v)

    if eval_['cm'] is not None:
        msg += '\n'
        msg += __spc + '+--------------------------------------+\n'
        msg += __spc + '| Confusion Matrix (Actuals along row) |\n'
        msg += __spc + '+--------------------------------------+\n'
        msg += hlp.pandas_repr(eval_['cm'], display__width=200,
                    precision=4, display__colheader_justify='right')
        msg += '\n\n' + __spc + 'Scaled Along row:\n'
        msg += __spc + '-----------------\n'
        msg += hlp.pandas_repr(eval_['cms'], display__width=200,
                    precision=4,
                    display__colheader_justify='right')
    msg += '\n\n'

    return msg


def banner(model_name):

    banner = (
            '\n' +
            __spc + '+' + '-' * (4 + len(model_name)) + '+\n' +
            __spc + '|' + ' ' * 2 + model_name + ' ' *2 + '|\n' +
            __spc + '+' + '-' * (4 + len(model_name)) + '+\n'
            )
    return banner


