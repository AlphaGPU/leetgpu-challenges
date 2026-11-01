import jax
import jax.numpy as jnp


# input are tensors on device
@jax.jit
def solve(input: jax.Array, M: int, N: int) -> jax.Array:
    # return output tensor directly
    pass
