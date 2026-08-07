import jax
import jax.numpy as jnp


# Q, K, V, alpha, beta are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    batch: int,
    seq_len: int,
    d: int,
) -> jax.Array:
    # return output tensor directly
    pass
