import jax
import jax.numpy as jnp


# Q, K, V, cos, sin are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
