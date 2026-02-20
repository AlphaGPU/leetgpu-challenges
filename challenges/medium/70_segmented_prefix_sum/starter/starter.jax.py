import jax
import jax.numpy as jnp


# values, flags are tensors on the GPU
@jax.jit
def solve(values: jax.Array, flags: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    pass
