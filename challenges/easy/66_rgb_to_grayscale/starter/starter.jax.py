import jax
import jax.numpy as jnp


# input is a tensor on GPU
@jax.jit
def solve(input: jax.Array, width: int, height: int) -> jax.Array:
    # return output tensor directly
    pass
