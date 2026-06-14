import jax
import jax.numpy as jnp


# X, gamma, beta are tensors on device
@jax.jit
def solve(
    X: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    N: int,
    C: int,
    H: int,
    W: int,
    G: int,
    eps: float,
) -> jax.Array:
    # return output tensor directly
    pass
