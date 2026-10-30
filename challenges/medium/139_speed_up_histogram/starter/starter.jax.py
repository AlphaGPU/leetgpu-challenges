import jax
import jax.numpy as jnp


# input are tensors on device
@jax.jit
def solve(input: jax.Array, N: int, num_bins: int) -> jax.Array:
    # return output tensor directly
    pass
