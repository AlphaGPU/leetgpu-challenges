import jax
import jax.numpy as jnp


# Q, K_int8, V_int8, k_scale, v_scale are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K_int8: jax.Array,
    V_int8: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
