import jax
import jax.numpy as jnp


# A, B, C are tensors on device
@jax.jit
def solve(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
) -> jax.Array:
    # return output tensor directly
    pass
