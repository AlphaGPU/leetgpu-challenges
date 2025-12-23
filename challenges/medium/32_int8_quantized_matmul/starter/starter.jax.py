import jax
import jax.numpy as jnp

# A, B are tensors on the GPU
@jax.jit
def solve(A: jax.Array, B: jax.Array, M: int, N: int, K: int, scale_A: float, scale_B: float, scale_C: float, zero_point_A: int, zero_point_B: int, zero_point_C: int) -> jax.Array:
    # return output tensor directly
    pass
