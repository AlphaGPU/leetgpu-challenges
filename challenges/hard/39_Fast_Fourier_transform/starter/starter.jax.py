import jax
import jax.numpy as jnp


# signal is a tensor on device
@jax.jit
def solve(signal: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
