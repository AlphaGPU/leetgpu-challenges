import jax
import jax.numpy as jnp


# u, delta, A, B, C, skip are tensors on GPU
@jax.jit
def solve(
    u: jax.Array,
    delta: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    skip: jax.Array,
    batch: int,
    seq_len: int,
    d_model: int,
    d_state: int,
) -> jax.Array:
    # return output tensor directly
    pass
