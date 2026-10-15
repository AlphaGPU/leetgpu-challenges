import jax
import jax.numpy as jnp


# x_q, x_scales, w_q, w_scales are tensors on device
@jax.jit
def solve(
    x_q: jax.Array,
    x_scales: jax.Array,
    w_q: jax.Array,
    w_scales: jax.Array,
    M: int,
    N: int,
    K: int,
) -> jax.Array:
    # return y tensor directly
    pass
