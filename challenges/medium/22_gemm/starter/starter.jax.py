import jax
import jax.numpy as jnp

# A, B are tensors on the GPU
@jax.jit
def solve(A: jax.Array, B: jax.Array, M: int, N: int, K: int, alpha: float, beta: float) -> jax.Array:
    # return output tensor directly
    pass
