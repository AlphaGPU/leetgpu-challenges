import jax
import jax.numpy as jnp


# x, weight, dy are tensors on device
@jax.jit
def solve(
    x: jax.Array, weight: jax.Array, dy: jax.Array, N: int, C: int, eps: float
) -> tuple[jax.Array, jax.Array]:
    # return output tensor directly
    pass
