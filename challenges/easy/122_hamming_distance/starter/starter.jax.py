import jax
import jax.numpy as jnp


# codes is a tensor on device
@jax.jit
def solve(codes: jax.Array, query: int, N: int) -> jax.Array:
    # return output tensor directly
    pass
