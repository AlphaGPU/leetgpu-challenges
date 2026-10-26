import jax
import jax.numpy as jnp


# G, Buf are tensors on device
@jax.jit
def solve(G: jax.Array, Buf: jax.Array, M: int, N: int, momentum: float) -> jax.Array:
    # return output tensor directly
    pass
