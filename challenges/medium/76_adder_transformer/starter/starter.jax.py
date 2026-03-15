import jax
import jax.numpy as jnp


# prompts, weights are tensors on GPU
@jax.jit
def solve(prompts: jax.Array, weights: jax.Array, batch_size: int) -> jax.Array:
    # return output tensor directly
    pass
