import jax
import jax.numpy as jnp


# x, W, A, B, adapter_ids are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    W: jax.Array,
    A: jax.Array,
    B: jax.Array,
    adapter_ids: jax.Array,
    T: int,
    K: int,
    N: int,
    R: int,
    L: int,
    scale: float,
) -> jax.Array:
    # return output tensor directly
    pass
