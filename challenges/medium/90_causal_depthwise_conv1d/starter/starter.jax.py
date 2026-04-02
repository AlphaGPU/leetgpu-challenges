import jax
import jax.numpy as jnp


# x, weight, bias are tensors on GPU
@jax.jit
def solve(
    x: jax.Array, weight: jax.Array, bias: jax.Array, B: int, L: int, D: int, K: int
) -> jax.Array:
    # return output tensor directly
    pass
