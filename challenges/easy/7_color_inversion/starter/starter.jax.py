import jax
import jax.numpy as jnp

# image is a tensor on the GPU
@jax.jit
def solve(image: jax.Array, width: int, height: int) -> jax.Array:
    # return output tensor directly
    pass
