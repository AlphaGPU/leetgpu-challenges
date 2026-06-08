import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, N: int, M: int, K: int) -> jax.Array:
    # return output tensor directly
    pass
