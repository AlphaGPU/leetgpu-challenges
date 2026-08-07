import jax
import jax.numpy as jnp


# q, kv_cache, W_UK, W_UV are tensors on device
@jax.jit
def solve(
    q: jax.Array,
    kv_cache: jax.Array,
    W_UK: jax.Array,
    W_UV: jax.Array,
    num_heads: int,
    seq_len: int,
    kv_lora_rank: int,
    head_dim: int,
    rope_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
