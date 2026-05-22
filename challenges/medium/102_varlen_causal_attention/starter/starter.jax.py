import jax
import jax.numpy as jnp


# Q, K, V, cu_seqlens are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    cu_seqlens: jax.Array,
    T: int,
    d: int,
    S: int,
) -> jax.Array:
    # return output tensor directly
    pass
