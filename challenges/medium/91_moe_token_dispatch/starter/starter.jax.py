import jax
import jax.numpy as jnp


# x, expert_idx are tensors on GPU
@jax.jit
def solve(
    x: jax.Array,
    expert_idx: jax.Array,
    T: int,
    D: int,
    E: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array]:
    pass
