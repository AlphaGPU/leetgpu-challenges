import jax
import jax.numpy as jnp


# Q, K, V, X, W_g are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    X: jax.Array,
    W_g: jax.Array,
    M: int,
    d_model: int,
    h: int,
) -> jax.Array:
    # return output tensor directly
    pass
