import jax
import jax.numpy as jnp


# logits is a tensor on the GPU
# Return tuple of (topk_weights, topk_indices)
@jax.jit
def solve(logits: jax.Array, M: int, E: int, k: int) -> tuple[jax.Array, jax.Array]:
    pass
