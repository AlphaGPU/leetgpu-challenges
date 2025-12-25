import jax
import jax.numpy as jnp


# agents is a tensor on the GPU
@jax.jit
def solve(agents: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
