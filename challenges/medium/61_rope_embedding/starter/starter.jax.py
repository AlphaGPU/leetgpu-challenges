import jax
import jax.numpy as jnp


# Q, cos, sin are tensors on the GPU
@jax.jit
def solve(Q: jax.Array, cos: jax.Array, sin: jax.Array, M: int, D: int) -> jax.Array:
    # return output tensor directly
    pass
