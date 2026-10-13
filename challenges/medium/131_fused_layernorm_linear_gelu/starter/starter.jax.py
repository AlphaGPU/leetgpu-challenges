import jax
import jax.numpy as jnp


# x, gamma, beta, W, bias are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    W: jax.Array,
    bias: jax.Array,
    M: int,
    K: int,
    N: int,
    eps: float,
) -> jax.Array:
    # return output tensor directly
    pass
