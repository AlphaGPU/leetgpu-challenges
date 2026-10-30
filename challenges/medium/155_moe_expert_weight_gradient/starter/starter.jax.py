import jax
import jax.numpy as jnp


# X, dY, group_offsets are tensors on device
@jax.jit
def solve(
    X: jax.Array,
    dY: jax.Array,
    group_offsets: jax.Array,
    G: int,
    M_total: int,
    K: int,
    N: int,
) -> jax.Array:
    # return output tensor directly
    pass
