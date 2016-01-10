# -*- coding: utf-8 -*-
from __future__ import division
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.random import RandomState


def numpy_rng_instance(numpy_rng=None):
    """Generates `numpy` `RandomState` object

    Parameters
    ----------
    numpy_rng : {None, `RandomState`, int}, optional
        `numpy` random state or an integer seed

    Returns
    -------
    `RandomState`
        `numpy` `RandomState` object
    """
    if isinstance(numpy_rng, RandomState):
        return numpy_rng
    else:
        return RandomState(numpy_rng)


def theano_rng_instance(theano_rng=None):
    """Generates `theano` `RandomStreams` object

    Parameters
    ----------
    heano_rng : {None, `RandomStream`, int}, optional
        `theano` random stream or an integer seed

    Returns
    -------
    `RandomStream`
        `theano` RandomStream object
    """

    if theano_rng is None:
        numpy_rng = RandomState(None)
        return RandomStreams(numpy_rng.randint(2 ** 30))
    elif isinstance(theano_rng, RandomStreams):
        return theano_rng
    else:
        return RandomStreams(theano_rng)
