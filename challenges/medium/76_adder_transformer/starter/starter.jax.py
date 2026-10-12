import jax
import jax.numpy as jnp


# prompts, weights are tensors on device
@jax.jit
def solve(prompts: jax.Array, weights: jax.Array, batch_size: int) -> jax.Array:
    # return output tensor directly
    pass
