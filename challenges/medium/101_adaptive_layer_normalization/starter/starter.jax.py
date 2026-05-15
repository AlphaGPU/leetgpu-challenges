import jax
import jax.numpy as jnp


# X, scale, shift are tensors on GPU
@jax.jit
def solve(
    X: jax.Array,
    scale: jax.Array,
    shift: jax.Array,
    B: int,
    N: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
