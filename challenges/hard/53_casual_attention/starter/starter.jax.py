import jax
import jax.numpy as jnp

# Q, K, V are tensors on the GPU
@jax.jit
def solve(Q: jax.Array, K: jax.Array, V: jax.Array,
          M: int, d: int) -> jax.Array:
    # return output tensor directly
    pass
