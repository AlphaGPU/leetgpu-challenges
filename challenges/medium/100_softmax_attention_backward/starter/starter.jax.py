import jax
import jax.numpy as jnp


# Q, K, V, dO are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array, K: jax.Array, V: jax.Array, dO: jax.Array, M: int, N: int, d: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return output tensor directly
    pass
