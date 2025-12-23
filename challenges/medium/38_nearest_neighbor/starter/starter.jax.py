import jax
import jax.numpy as jnp

# points is a tensor on the GPU
@jax.jit
def solve(points: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
