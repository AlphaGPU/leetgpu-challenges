import jax
import jax.numpy as jnp


# x, w_values, w_meta are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    w_values: jax.Array,
    w_meta: jax.Array,
    M: int,
    N: int,
    K: int,
) -> jax.Array:
    # return output tensor directly
    pass
