import jax
import jax.numpy as jnp


# src, dst are tensors on GPU
@jax.jit
def solve(src: jax.Array, dst: jax.Array, N: int, M: int) -> jax.Array:
    # return output tensor directly
    pass
