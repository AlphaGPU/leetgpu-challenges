import jax
import jax.numpy as jnp


# x, W_qkv, cos_sin_cache, positions, K_cache, V_cache are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    W_qkv: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    K_cache: jax.Array,
    V_cache: jax.Array,
    B: int,
    d_model: int,
    H_q: int,
    H_kv: int,
    D: int,
    S_max: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return output tensor directly
    pass
