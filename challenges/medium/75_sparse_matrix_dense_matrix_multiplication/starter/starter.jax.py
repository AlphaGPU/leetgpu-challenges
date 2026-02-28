import jax
import jax.numpy as jnp


# A, B are tensors on GPU
@jax.jit
def solve(A: jax.Array, B: jax.Array, M: int, N: int, K: int, nnz: int) -> jax.Array:
    # return output tensor directly
    pass
