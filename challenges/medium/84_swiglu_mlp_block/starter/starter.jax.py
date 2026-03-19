import jax
import jax.numpy as jnp


# x, W_gate, W_up, W_down are tensors on GPU
@jax.jit
def solve(
    x: jax.Array,
    W_gate: jax.Array,
    W_up: jax.Array,
    W_down: jax.Array,
    M: int,
    d_model: int,
    d_ffn: int,
) -> jax.Array:
    # return output tensor directly
    pass
