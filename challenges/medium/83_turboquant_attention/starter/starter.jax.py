import jax
import jax.numpy as jnp


# Q, K_idx, Pi, codebook are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K_idx: jax.Array,
    Pi: jax.Array,
    codebook: jax.Array,
    B: int,
    S: int,
    D: int,
    C: int,
) -> jax.Array:
    # return output tensor directly
    pass
