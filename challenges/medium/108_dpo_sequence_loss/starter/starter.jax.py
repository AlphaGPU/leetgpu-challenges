import jax
import jax.numpy as jnp


# chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps are tensors on device
@jax.jit
def solve(
    chosen_logps: jax.Array,
    rejected_logps: jax.Array,
    chosen_ref_logps: jax.Array,
    rejected_ref_logps: jax.Array,
    beta: float,
    B: int,
) -> jax.Array:
    # return output tensor directly
    pass
