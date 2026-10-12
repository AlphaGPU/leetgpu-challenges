import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, B: int, N: int) -> jax.Array:
    # return output tensor directly
    pass
