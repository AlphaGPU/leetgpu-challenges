import jax
import jax.numpy as jnp


# src, index are tensors on device
@jax.jit
def solve(src: jax.Array, index: jax.Array, N: int, D: int, M: int) -> jax.Array:
    # return output tensor directly
    pass
