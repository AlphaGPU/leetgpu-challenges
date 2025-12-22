import jax
import jax.numpy as jnp

# logits, true_labels are tensors on the GPU
@jax.jit
def solve(logits: jax.Array, true_labels: jax.Array, N: int, C: int):
    pass
