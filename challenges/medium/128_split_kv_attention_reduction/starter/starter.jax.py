import jax
import jax.numpy as jnp


# partial_out, partial_lse are tensors on device
@jax.jit
def solve(
    partial_out: jax.Array,
    partial_lse: jax.Array,
    num_splits: int,
    num_heads: int,
    head_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
