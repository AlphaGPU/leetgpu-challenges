import jax
import jax.numpy as jnp


# x, residual, gamma, beta are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    residual: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    N: int,
    C: int,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # return (y, residual_out, mean, rstd) directly
    pass
