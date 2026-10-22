import jax
import jax.numpy as jnp


# x, gamma, W, w_scale are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    gamma: jax.Array,
    W: jax.Array,
    w_scale: jax.Array,
    M: int,
    N: int,
    K: int,
) -> jax.Array:
    # return output tensor directly
    pass
