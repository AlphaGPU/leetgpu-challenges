import jax
import jax.numpy as jnp


# Q, K, V are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    cache_len: int,
    head_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
