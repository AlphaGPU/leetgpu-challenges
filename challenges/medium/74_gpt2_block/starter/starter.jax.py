import jax
import jax.numpy as jnp


# x, weights are tensors on the GPU
@jax.jit
def solve(x: jax.Array, weights: jax.Array, seq_len: int) -> jax.Array:
    # return output tensor directly
    pass
