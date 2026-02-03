import jax
import jax.numpy as jnp


# logits is a tensor on the GPU
@jax.jit
def solve(logits: jax.Array, M: int, E: int, k: int) -> tuple[jax.Array, jax.Array]:
    pass
