import logging
#import theano
from theano import tensor as T

from ..common.random import numpy_rng_instance
from ..common.random import theano_rng_instance
from ..arch.feed_forward import FeedForwardNetwork as Network
from . import misc as hlp
from .. import models

logger = logging.getLogger(__name__)


def compile_model(arch_descr, data, cost_func,
        confidence_func=None, loss_func=None, predict_func=None,
        scoring_func=None, model_class='BaseSupervisedModel',
        numpy_rng=None, theano_rng=None, gd_params=None):

    logger.info('setting up the random number generators')
    numpy_rng = numpy_rng_instance(numpy_rng)
    theano_rng = theano_rng_instance(theano_rng)

    logger.info('Constructing the architecture')
    input_type = hlp.get_tensor_type(data['train'][0])
    X = getattr(T, input_type)('X', dtype=str(data['train'][0].dtype))
    arch = Network(
            X, arch_descr, numpy_rng=numpy_rng, theano_rng=theano_rng
            )

    logger.info('Setting up the model parameters')
    output_type = hlp.get_tensor_type(data['train'][1])
    output_dtype = str(data['train'][1].dtype)
    ModelClass = getattr(models, model_class)
    cost_func = hlp.construct_function(cost_func)
    confidence_func = hlp.construct_function(confidence_func)
    loss_func = hlp.construct_function(loss_func)
    scoring_func = hlp.construct_function(scoring_func)
    predict_func = hlp.construct_function(predict_func)
    model_params = dict(
        cost_func=cost_func,
        loss_func=loss_func,
        predict_func=predict_func,
        scoring_func=scoring_func,
        confidence_func=confidence_func,
        output_type=output_type,
        output_dtype=output_dtype
        )
    gd_params = {} if gd_params is None else gd_params
    model_params.update(gd_params)

    logger.info('Compiling the model')
    model = ModelClass(arch, **model_params)

    return model

