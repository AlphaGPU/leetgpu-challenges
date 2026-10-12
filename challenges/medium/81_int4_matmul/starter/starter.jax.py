import jax
import jax.numpy as jnp


# x, w_q, scales are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    M: int,
    N: int,
    K: int,
    group_size: int,
) -> jax.Array:
    # return output tensor directly
    pass
