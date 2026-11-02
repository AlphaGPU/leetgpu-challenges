import jax
import jax.numpy as jnp


# q, k, v are tensors on device
@jax.jit
def solve(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    B: int,
    H: int,
    H_KV: int,
    S: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
