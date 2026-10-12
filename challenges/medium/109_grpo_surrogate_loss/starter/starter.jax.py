import jax
import jax.numpy as jnp


# rewards, log_pi, log_pi_old, log_ref are tensors on device
@jax.jit
def solve(
    rewards: jax.Array,
    log_pi: jax.Array,
    log_pi_old: jax.Array,
    log_ref: jax.Array,
    clip_eps: float,
    beta: float,
    B: int,
    G: int,
    S: int,
) -> jax.Array:
    # return output tensor directly
    pass
