import jax
import jax.numpy as jnp


# Q, K, V are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    window_size: int,
    softcap: float,
) -> jax.Array:
    # return output tensor directly
    pass
