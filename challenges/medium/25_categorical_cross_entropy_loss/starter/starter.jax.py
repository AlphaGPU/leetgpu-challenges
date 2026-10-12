import jax
import jax.numpy as jnp


# logits, true_labels are tensors on device
@jax.jit
def solve(logits: jax.Array, true_labels: jax.Array, N: int, C: int) -> jax.Array:
    # return output tensor directly
    pass
