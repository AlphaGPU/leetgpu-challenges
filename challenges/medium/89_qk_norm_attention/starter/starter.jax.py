import jax
import jax.numpy as jnp


# Q, K, V, q_weight, k_weight are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    N: int,
    d_model: int,
    h: int,
    eps: float,
) -> jax.Array:
    # return output tensor directly
    pass
