# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np

import theano
from theano import tensor as T, shared


def sgd_updates(grads, params, eta_0=0.13):

    updates = [
        (p, p - eta_0 * grad) for p, grad in zip(params, grads)
    ]
    return updates


def sgd_updates_momentum(grads, params, eta_0=0.13, mu=0.5):

    deltas = []
    for p in params:
        deltas.append(
            shared(
                np.zeros(p.get_value(borrow=True).shape,
                         dtype=theano.config.floatX),
                name='delta_' + p.name
            )
        )

    new_deltas = [
        - eta_0 * grad + mu * delta
        for grad, delta in zip(grads, deltas)
    ]

    updates = [
        (p, p + delta) for p, delta in zip(params, new_deltas)
    ]

    updates += [
        (delta, new_delta) for delta, new_delta in zip(deltas, new_deltas)
    ]

    return updates


def adadelta(grads, params, eta_0=1.0, rho=0.5, eps=1e-8):

    return sgd_updates_adadelta(
            grads, params, eta_0, rho_delta=rho, rho_grad=rho, eps=eps
            )


def sgd_updates_adadelta(grads, params, eta_0=1.0, rho_delta=0.5, rho_grad=0.5,
                         eps=1e-8):
    grad_ss = []
    delta_ss = []

    for p in params:
        grad_ss.append(
            shared(
                np.zeros(p.get_value(borrow=True).shape),
                name='grad_wrt_' + p.name + '_ss'
            )
        )

        delta_ss.append(
            shared(
                np.zeros(p.get_value(borrow=True).shape),
                name='delta_' + p.name + '_ss'
            )
        )

    new_grad_ss = [
        rho_grad * grad**2 + (1 - rho_grad) * ss
        for grad, ss in zip(grads, grad_ss)
    ]

    deltas = [
        - eta_0 * grad * T.sqrt((d_ss + eps) / (g_ss + eps))
        for grad, g_ss, d_ss in zip(grads, new_grad_ss, delta_ss)
    ]

    new_delta_ss = [
        rho_delta * delta**2 + (1 - rho_delta) * ss
        for delta, ss in zip(deltas, delta_ss)
    ]

    updates = [
        (p, p + delta) for p, delta in zip(params, deltas)
    ]

    updates += [
        (ss, new_ss) for ss, new_ss in zip(grad_ss, new_grad_ss)
    ]

    updates += [
        (ss, new_ss) for ss, new_ss in zip(delta_ss, new_delta_ss)
    ]

    return updates
