import jax
import jax.numpy as jnp


# x, W, A, B are tensors on GPU
@jax.jit
def solve(
    x: jax.Array,
    W: jax.Array,
    A: jax.Array,
    B: jax.Array,
    batch: int,
    d_in: int,
    d_out: int,
    rank: int,
    lora_scale: float,
) -> jax.Array:
    # return output tensor directly
    pass
