import jax
import jax.numpy as jnp

# Q, K, V are tensors on the GPU
@jax.jit
def solve(Q: jax.Array, K: jax.Array, V: jax.Array,
          N: int, d_model: int, h: int) -> jax.Array:
    # return output tensor directly
    pass
