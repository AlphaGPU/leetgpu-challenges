import jax
import jax.numpy as jnp


# logits are tensors on GPU
@jax.jit
def solve(logits: jax.Array, min_p: float, B: int, V: int) -> jax.Array:
    # return output tensor directly
    pass
