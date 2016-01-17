import logging
from collections import OrderedDict
# import inspect

from ..common.random import numpy_rng_instance
from ..common.random import theano_rng_instance
from ..common import metrics
from ..common import eval_metrics
from ..common.eval_metrics import confusion_matrix
from ..misc.viz import trace
from .loaders import read_file

def evaluation(data, model, eval_metrics=[], confusion=False,
               confusion_kwrgs={}):

    pred = model.predict(data[0])
    eval_ = []
    for m in eval_metrics:
        k = m['name']
        f = construct_eval_metrics(m)
        eval_.append((k, f(pred, data[1])))
    eval_ = OrderedDict(eval_)

    if confusion:
        cm = confusion_matrix(pred, data[1])
        cms = confusion_matrix(pred, data[1], scaled=True)
    else:
        cm = None
        cms = None
    return {'eval_metrics':eval_, 'cm':cm, 'cms':cms}


def visualize(model_pickle, i=1, figsize=(12,6)):

    model = read_file(model_pickle, 'pkl')
    fig = trace(model, i=i, figsize=figsize)
    fig.suptitle(model_pickle[:-4])
    return fig


def setup_basic_logging(log_file=None, level=logging.INFO,
                        format='  %(asctime)s => %(message)s'):
    logging.basicConfig(
            filename=log_file, level=level, format=format
            )


def construct_eval_metrics(func_descr):
    return construct_function(func_descr, module=eval_metrics)


def construct_function(func_descr, module=metrics):
    if func_descr is None:
        return func_descr
    elif hasattr(func_descr, '__call__'):
        return func_descr

    func = getattr(module, func_descr.pop('name'))

    f = lambda *Yy: func(*Yy, **func_descr)

    return f

def initialize_rngs(numpy_rng_seed=None, theano_rng_seed=None):
    numpy_rng = numpy_rng_instance(numpy_rng_seed)
    theano_rng = theano_rng_instance(theano_rng_seed)
    return numpy_rng, theano_rng


def get_tensor_type(data):

    if data.ndim == 1:
        type_ = 'vector'
    elif data.ndim == 2:
        type_ = 'matrix'
    else:
        raise ValueError(
                'Data of dimension %i is not yet supported' % data.ndim
                )
    return type_


def pandas_repr(mat, col_names=None, row_names=None, margin=2, **pd_options):
    import pandas as pd
    for k, v in pd_options.items():
        k = k.replace('__', '.')
        pd.set_option(k, v)
    df = pd.DataFrame(data=mat, index=row_names, columns=col_names)
    return repr(df).replace('\n', '\n' + ' ' * margin)
