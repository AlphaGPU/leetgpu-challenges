import jax
import jax.numpy as jnp

# A, B are tensors on the GPU
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
