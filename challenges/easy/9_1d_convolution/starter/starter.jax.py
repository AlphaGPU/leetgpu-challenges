import jax
import jax.numpy as jnp

# input, kernel are tensors on the GPU
@jax.jit
def solve(input: jax.Array, kernel: jax.Array, input_size: int, kernel_size: int):
    pass
