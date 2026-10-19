import jax
import jax.numpy as jnp


# x, W_gate_up are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    W_gate_up: jax.Array,
    M: int,
    H: int,
    I: int,
) -> jax.Array:
    # return output tensor directly
    pass
