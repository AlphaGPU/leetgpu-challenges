import jax
import jax.numpy as jnp


# Q, K_prefix, V_prefix, K_suffix, V_suffix are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K_prefix: jax.Array,
    V_prefix: jax.Array,
    K_suffix: jax.Array,
    V_suffix: jax.Array,
    batch_size: int,
    num_heads: int,
    prefix_len: int,
    suffix_len: int,
    head_dim: int,
) -> jax.Array:
    # return output tensor directly
    pass
