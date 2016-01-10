import logging
import pickle
# import inspect

from ..common.random import numpy_rng_instance
from ..common.random import theano_rng_instance
from ..common import metrics
from ..common import eval_metrics


def construct_eval_metrics(func_descr):
    return construct_function(func_descr, module=eval_metrics)


def construct_function(func_descr, module=metrics):
    if func_descr is None:
        return func_descr
    func = getattr(module, func_descr.pop('name'))

    f = lambda *Yy: func(*Yy, **func_descr)

    return f


def read_file(filepath, mode='r'):
    with open(filepath, mode) as f:
        txt = f.read()
    return txt


def setup_basic_logging(log_file=None, level=logging.INFO,
                        format='%(asctime)s => %(message)s'):
    logging.basicConfig(
            filename=log_file, level=level, format=format
            )


def load_datasets(filepath):
    with open(filepath, 'rb') as f:
        datasets = pickle.load(f)

    if isinstance(datasets, list):
        training_data, eval_data = datasets
        return {'training': training_data, 'eval': eval_data}
    else:
        training_data = datasets
        return {'training': training_data}


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


def pandas_repr(mat, col_names=None, row_names=None, **pd_options):
    import pandas as pd
    for k, v in pd_options.items():
        k = k.replace('__', '.')
        pd.set_option(k, v)
    df = pd.DataFrame(data=mat, index=row_names, columns=col_names)
    return repr(df)
