import jax
import jax.numpy as jnp


# x, dt, dt_bias, A, B, C, D, z are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    dt: jax.Array,
    dt_bias: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
    z: jax.Array,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_groups: int,
    state_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
