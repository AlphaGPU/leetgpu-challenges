import jax
import jax.numpy as jnp


# image is a tensor on GPU
@jax.jit
def solve(
    image: jax.Array, H: int, W: int, spatial_sigma: float, range_sigma: float, radius: int
) -> jax.Array:
    # return output tensor directly
    pass
