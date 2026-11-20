import jax
import jax.numpy as jnp


# Q, K, V, O, dO, lse are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    O: jax.Array,
    dO: jax.Array,
    lse: jax.Array,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return output tensor directly
    pass
