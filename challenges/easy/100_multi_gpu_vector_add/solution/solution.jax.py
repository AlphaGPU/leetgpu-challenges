import jax
import jax.numpy as jnp


# JAX multi-GPU: the runner calls jax.distributed.initialize before loading this
# module, so jax.process_count() / jax.process_index() reflect the world.
# Inputs are fully replicated across ranks (deterministic seeding). This trivial
# solution computes the full result locally on each rank — cross-rank collectives
# via jax.experimental.multihost_utils are also supported.
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    return A + B
