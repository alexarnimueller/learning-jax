from typing import Any, Callable, NamedTuple, Optional, Tuple

import numpy as np

import jax.numpy as jnp
from jax.tree_util import tree_map

PyTree = Any


class Optimizer(NamedTuple):
    # Given the parameters, initialize any optimizer state as tuple
    init: Callable[[PyTree], tuple]
    # Given the gradients, optimizer state, and evt. parameters, return
    # the parameter updates and new optimizer state
    update: Callable[[PyTree, tuple, Optional[PyTree]], Tuple[PyTree, tuple]]


def sgd(lr):
    def init(params):
        return tuple()

    def update(updates, state, params=None):
        updates = tree_map(lambda u: -lr * u, updates)
        return updates, state

    return Optimizer(init, update)


def sgd_momentum(lr, momentum=0.0):
    def init(params):
        param_momentum = tree_map(jnp.zeros_like, params)
        return param_momentum

    def update(updates, state, params=None):
        state = tree_map(lambda m, g: (1 - momentum) * g + momentum * m, state, updates)
        updates = tree_map(lambda m: -lr * m, state)
        return updates, state

    return Optimizer(init, update)


def adam(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    def init(params):
        step = 0.0
        param_momentum = tree_map(jnp.zeros_like, params)
        param_2nd_momentum = tree_map(jnp.zeros_like, params)
        return (step, param_momentum, param_2nd_momentum)

    def update(updates, state, params=None):
        # Update momentum and adapt. lr
        step, param_momentum, param_2nd_momentum = state
        step += 1
        param_momentum = tree_map(
            lambda m, g: (1 - beta1) * g + beta1 * m, param_momentum, updates
        )
        param_2nd_momentum = tree_map(
            lambda m2, g: (1 - beta2) * g**2 + beta2 * m2, param_2nd_momentum, updates
        )

        def update_param(m, m2):  # Calculate update for single parameter
            # Bias correction
            m /= 1 - beta1**step
            m2 /= 1 - beta2**step
            return -m * lr / (jnp.sqrt(m2) + eps)

        # Update for all parameters
        updates = tree_map(update_param, param_momentum, param_2nd_momentum)
        return updates, (step, param_momentum, param_2nd_momentum)

    return Optimizer(init, update)


def cosine_warmup_schedule(base_lr: float, warmup: int, max_iters: int):
    assert warmup > 0 and max_iters > 0

    # Create function to return lr based on iteration count
    def get_lr(train_iter):
        lr_factor = 0.5 * (1 + np.cos(np.pi * train_iter / max_iters))
        if train_iter <= warmup:
            lr_factor *= train_iter * 1.0 / warmup
        return lr_factor * base_lr

    return get_lr
