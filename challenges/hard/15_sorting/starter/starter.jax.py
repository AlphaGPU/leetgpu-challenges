import jax
import jax.numpy as jnp


# data is a tensor on device
@jax.jit
def solve(data: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
