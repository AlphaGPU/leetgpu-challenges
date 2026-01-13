import jax
import jax.numpy as jnp


# X, S are tensors on the GPU
@jax.jit
def solve(X: jax.Array, S: jax.Array, M: int, N: int, TILE_SIZE: int) -> jax.Array:
    # return output tensor Y directly
    pass
