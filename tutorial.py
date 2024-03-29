import time

from IPython.display import set_matplotlib_formats

import jax
import jax.numpy as jnp

set_matplotlib_formats("svg", "pdf")

print("Using jax", jax.__version__)

# zeros, showing types etc.
a = jnp.zeros((2, 5), dtype=jnp.float32)
print(a)
print(type(a))
print(a.devices())

# range
b = jnp.arange(6)
print(b)

# getting array from GPU to CPU
b_cpu = jax.device_get(b)
print(b_cpu.__class__)

# putting array back to GPU
b_gpu = jax.device_put(b_cpu)
print(f"Device put: {b_gpu.__class__} on {b_gpu.devices()}")
print(b_cpu + b_gpu)

# in place modifications not possible
b_new = b.at[0].set(1)
print("Original array:", b)
print("Changed array:", b_new)

# randomness
rng = jax.random.PRNGKey(42)  # PRNG state

# A non-desirable way of generating pseudo-random numbers... (the same)
jax_random_number_1 = jax.random.normal(rng)
jax_random_number_2 = jax.random.normal(rng)
print("JAX - Random number 1:", jax_random_number_1)
print("JAX - Random number 2:", jax_random_number_2)

# create individual keys to get different random numbers
rng, subkey1, subkey2 = jax.random.split(rng, num=3)
jax_random_number_1 = jax.random.normal(subkey1)
jax_random_number_2 = jax.random.normal(subkey2)
print("JAX new - Random number 1:", jax_random_number_1)
print("JAX new - Random number 2:", jax_random_number_2)


# We want to write our main code of JAX in functions that do not affect anything else besides its outputs
# Only the network execution, which we want to do very efficiently on our accelerator (GPU or TPU),
# should strictly follow these constraints. -> don’t write code with side-effects


def simple_graph(x):
    x = x + 2
    x = x**2
    x = x + 3
    y = x.mean()
    return y


inp = jnp.arange(3, dtype=jnp.float32)
print("Input", inp)
print("Output", simple_graph(inp))

# to represent the function in jaxpr:
jax.make_jaxpr(simple_graph)(inp)
# Var* are constants and Var+ are input arguments

# gradient computation outputs a function that lets you obtain the gradient from the input data
grad_function = jax.grad(simple_graph)
gradients = grad_function(inp)
print("Gradient", gradients)
jax.make_jaxpr(grad_function)(inp)

# to also get the actual output of the function:
val_grad_function = jax.value_and_grad(simple_graph)
val, grad = val_grad_function(inp)
print(val, grad)

# To compile a function just in time (jit), JAX provides the jax.jit transformation
jitted_function = jax.jit(simple_graph)
rng, normal_rng = jax.random.split(rng)
large_input = jax.random.normal(normal_rng, (1000,))

# Run the jitted function once to start compilation based on this input shape
_ = jitted_function(large_input)

# measure time
start = time.process_time()
jitted_function(large_input).block_until_ready()
print("JIT:", time.process_time() - start)
start = time.process_time()
simple_graph(large_input).block_until_ready()
print("Non-JIT:", time.process_time() - start)

# same works for gradient calculation:
jitted_grad_function = jax.jit(grad_function)
_ = jitted_grad_function(large_input)  # Apply once to compile
start = time.process_time()
grad_function(large_input).block_until_ready()
print("JIT:", time.process_time() - start)
start = time.process_time()
jitted_grad_function(large_input).block_until_ready()
print("Non-JIT:", time.process_time() - start)
