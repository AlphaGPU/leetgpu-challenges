import jax
import jax.numpy as jnp


# A is a tensor on GPU
@jax.jit
def solve(A: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
