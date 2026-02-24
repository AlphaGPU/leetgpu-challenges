import jax
import jax.numpy as jnp


# dist is a tensor on the GPU
@jax.jit
def solve(dist: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
