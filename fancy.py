# Automatic Vectorization with vmap

import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(42)


def simple_linear(x, w, b):
    # We could already vectorize this function with matmul, but as an example,
    # let us use a non-vectorized function with same output
    return (x[:, None] * w).sum(axis=0) + b


# Example inputs, not batched
rng, x_rng, w_rng, b_rng = jax.random.split(rng, 4)
x_in = jax.random.normal(x_rng, (4,))
w_in = jax.random.normal(w_rng, (4, 3))
b_in = jax.random.normal(b_rng, (3,))

simple_linear(x_in, w_in, b_in)

# if we now would like to add a batch dimension, we would need to redefine the above function.
# but with vmap we can automatically vectorize the function to also take batches:
vectorized_linear = jax.vmap(
    simple_linear,
    in_axes=(0, None, None),  # Which axes to vectorize for each input
    out_axes=0,  # Which axes to map to in the output
)

# run batched example:
x_vec_in = jnp.stack([x_in] * 5, axis=0)
vectorized_linear(x_vec_in, w_in, b_in)

# for parallell execution on multiple devices, check https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
