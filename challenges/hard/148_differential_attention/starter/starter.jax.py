import jax
import jax.numpy as jnp


# Q, K, V, lambdas are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    lambdas: jax.Array,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    lambda_init: float,
) -> jax.Array:
    # return output tensor directly
    pass
