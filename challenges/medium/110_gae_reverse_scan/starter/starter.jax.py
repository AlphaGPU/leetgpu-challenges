import jax
import jax.numpy as jnp


# rewards, values are tensors on device
@jax.jit
def solve(
    rewards: jax.Array,
    values: jax.Array,
    gamma: float,
    lam: float,
    B: int,
    S: int,
) -> jax.Array:
    # return output tensor directly
    pass
