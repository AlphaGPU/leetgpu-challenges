import jax
import jax.numpy as jnp

# predictions, targets are tensors on the GPU
@jax.jit
def solve(predictions: jax.Array, targets: jax.Array, N: int):
    pass
