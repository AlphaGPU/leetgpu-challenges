import jax
import jax.numpy as jnp


# x, weights are tensors on GPU
@jax.jit
def solve(x: jax.Array, weights: jax.Array, seq_len: int) -> jax.Array:
    # return output tensor directly
    pass
