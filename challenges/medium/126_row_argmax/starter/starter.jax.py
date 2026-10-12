import jax
import jax.numpy as jnp


# logits is a tensor on device
@jax.jit
def solve(logits: jax.Array, B: int, V: int) -> jax.Array:
    # return output tensor directly
    pass
