import jax
import jax.numpy as jnp


# image is a tensor on the GPU
@jax.jit
def solve(image: jax.Array, H: int, W: int, H_out: int, W_out: int) -> jax.Array:
    # return output tensor directly
    pass
