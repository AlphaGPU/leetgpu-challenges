import jax
import jax.numpy as jnp


# positions, masses are tensors on the GPU
# positions shape: (N, 3), masses shape: (N,)
@jax.jit
def solve(positions: jax.Array, masses: jax.Array, N: int) -> jax.Array:
    # return accelerations array of shape (N, 3)
    pass
