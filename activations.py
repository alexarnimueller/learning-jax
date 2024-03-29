import math

import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp


class Sigmoid(nn.Module):
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))


class Tanh(nn.Module):
    def __call__(self, x):
        x_exp, neg_x_exp = jnp.exp(x), jnp.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)


class ReLU(nn.Module):
    def __call__(self, x):
        return jnp.maximum(x, 0)


class LeakyReLU(nn.Module):
    alpha: float = 0.1

    def __call__(self, x):
        return jnp.where(x > 0, x, self.alpha * x)


class ELU(nn.Module):
    def __call__(self, x):
        return jnp.where(x > 0, x, jnp.exp(x) - 1)


class Swish(nn.Module):
    def __call__(self, x):
        return x * nn.sigmoid(x)


def get_grads(act_fn, x):
    """
    Computes the gradients of an activation function at specified positions.

    Inputs:
        act_fn - An module or function of the forward pass of the activation function.
        x - 1D input array.
    Output:
        An array with the same size of x containing the gradients of act_fn at x.
    """
    return jax.vmap(jax.grad(act_fn))(x)


def vis_act_fn(act_fn, ax, x):
    # Run activation function
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    # Push x, y and gradients back to cpu for plotting
    # x, y, y_grads = x.cpu().numpy(), y.cpu().numpy(), y_grads.cpu().numpy()
    # Plotting
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grads, linewidth=2, label="Gradient")
    ax.set_title(act_fn.__class__.__name__)
    ax.legend()
    ax.set_ylim(-1.5, x.max())


act_fn_by_name = {
    "identity": lambda x: x,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
}


if __name__ == "__main__":
    # Add activation functions if wanted
    act_fns = [act_fn() for n, act_fn in act_fn_by_name.items() if n != "identity"]
    x = np.linspace(
        -5, 5, 1000
    )  # Range on which we want to visualize the activation functions
    # Plotting
    rows = math.ceil(len(act_fns) / 2.0)
    fig, ax = plt.subplots(rows, 2, figsize=(8, rows * 4))
    for i, act_fn in enumerate(act_fns):
        vis_act_fn(act_fn, ax[divmod(i, 2)], x)
    fig.subplots_adjust(hspace=0.3)
    plt.savefig("./plots/activations.png")
