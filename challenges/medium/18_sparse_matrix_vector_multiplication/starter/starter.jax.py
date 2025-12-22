import jax
import jax.numpy as jnp

# A, x are tensors on the GPU
@jax.jit
def solve(A: jax.Array, x: jax.Array, M: int, N: int, nnz: int) -> jax.Array:
    # return output tensor directly
    pass
