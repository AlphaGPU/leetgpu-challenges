import jax
import jax.numpy as jnp


# x, gamma, mean, rstd, dy are tensors on device
@jax.jit
def solve(
    x: jax.Array, gamma: jax.Array, mean: jax.Array, rstd: jax.Array, dy: jax.Array, N: int, C: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return output tensor directly
    pass
