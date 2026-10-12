import jax
import jax.numpy as jnp


# a, x are tensors on device
@jax.jit
def solve(a: jax.Array, x: jax.Array, B: int, L: int) -> jax.Array:
    # return output tensor directly
    pass
