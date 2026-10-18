import jax
import jax.numpy as jnp


# Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K_cache: jax.Array,
    V_cache: jax.Array,
    block_table: jax.Array,
    context_lens: jax.Array,
    cu_seqlens_q: jax.Array,
    num_seqs: int,
    total_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
) -> jax.Array:
    # return output tensor directly
    pass
