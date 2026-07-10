import jax
import jax.numpy as jnp


# x, W_qkv are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    W_qkv: jax.Array,
    M: int,
    num_heads: int,
    head_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return output tensors directly
    pass
