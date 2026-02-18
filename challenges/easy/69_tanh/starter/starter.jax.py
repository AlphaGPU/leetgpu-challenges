import jax
import jax.numpy as jnp


# X is a tensor on GPU
@jax.jit
def solve(X: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
