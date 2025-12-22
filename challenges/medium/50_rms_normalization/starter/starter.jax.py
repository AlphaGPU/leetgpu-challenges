import jax
import jax.numpy as jnp

# input, gamma, beta are tensors on the GPU
@jax.jit
def solve(input: jax.Array, gamma: jax.Array, beta: jax.Array, 
          N: int, eps: float) -> jax.Array:
    # return output tensor directly
    pass
