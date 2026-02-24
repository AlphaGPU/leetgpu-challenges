import jax
import jax.numpy as jnp


# x, ln1_weight, ln1_bias, W_qkv, b_qkv, W_attn_proj, b_attn_proj, ln2_weight, ln2_bias, W_fc, b_fc, W_proj, b_proj are tensors on the GPU
@jax.jit
def solve(
    x: jax.Array,
    ln1_weight: jax.Array,
    ln1_bias: jax.Array,
    W_qkv: jax.Array,
    b_qkv: jax.Array,
    W_attn_proj: jax.Array,
    b_attn_proj: jax.Array,
    ln2_weight: jax.Array,
    ln2_bias: jax.Array,
    W_fc: jax.Array,
    b_fc: jax.Array,
    W_proj: jax.Array,
    b_proj: jax.Array,
    seq_len: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
