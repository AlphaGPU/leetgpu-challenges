import jax
import jax.numpy as jnp


# Q, K, V are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int,
    num_selected: int,
) -> jax.Array:
    # return output tensor directly
    pass
