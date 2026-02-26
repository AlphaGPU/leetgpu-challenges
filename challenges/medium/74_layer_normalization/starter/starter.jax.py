import jax
import jax.numpy as jnp


# input, weight, bias are tensors on the GPU
@jax.jit
def solve(
    input: jax.Array, weight: jax.Array, bias: jax.Array, N: int, C: int, eps: float
) -> jax.Array:
    # return output tensor directly
    pass
