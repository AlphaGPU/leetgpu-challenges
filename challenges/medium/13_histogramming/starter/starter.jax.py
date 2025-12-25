import jax
import jax.numpy as jnp


# input is a tensor on the GPU
@jax.jit
def solve(input: jax.Array, N: int, num_bins: int) -> jax.Array:
    # return output tensor directly
    pass
