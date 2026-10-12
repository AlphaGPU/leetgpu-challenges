import jax
import jax.numpy as jnp


# positions, masses are tensors on GPU
@jax.jit
def solve(positions: jax.Array, masses: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
