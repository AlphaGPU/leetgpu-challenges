import jax
import jax.numpy as jnp


# A, B, group_offsets are tensors on GPU
@jax.jit
def solve(
    A: jax.Array,
    B: jax.Array,
    group_offsets: jax.Array,
    G: int,
    M_total: int,
    K: int,
    N: int,
) -> jax.Array:
    # return output tensor directly
    pass
