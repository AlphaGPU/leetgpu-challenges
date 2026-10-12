import jax
import jax.numpy as jnp


# codes, query are tensors on device
@jax.jit
def solve(codes: jax.Array, query: jax.Array, N: int, k: int) -> jax.Array:
    # return output tensor directly
    pass
