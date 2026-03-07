import jax
import jax.numpy as jnp


# data is a tensor on the GPU
@jax.jit
def solve(data: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
