import jax
import jax.numpy as jnp


# q, k_cache, v_cache, idx_q, idx_k, idx_w are tensors on device
@jax.jit
def solve(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    idx_q: jax.Array,
    idx_k: jax.Array,
    idx_w: jax.Array,
    B: int,
    S: int,
    H: int,
    D: int,
    H_I: int,
    D_I: int,
    topk: int,
) -> jax.Array:
    # return output tensor directly
    pass
