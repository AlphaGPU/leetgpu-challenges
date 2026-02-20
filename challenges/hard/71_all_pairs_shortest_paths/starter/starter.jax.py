import jax
import jax.numpy as jnp


# dist is a tensor on the GPU (N*N floats, row-major)
@jax.jit
def solve(dist: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
