import jax
import jax.numpy as jnp


# Q, K, V are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    seq_len: int,
    d_model: int,
    gamma: float,
) -> jax.Array:
    # return output tensor directly
    pass
