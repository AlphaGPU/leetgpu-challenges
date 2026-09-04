import jax
import jax.numpy as jnp


# x, c, weights are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    c: jax.Array,
    weights: jax.Array,
    batch_size: int,
    seq_len: int,
) -> jax.Array:
    # return output tensor directly
    pass
