import jax
import jax.numpy as jnp


# beam_scores, token_logprobs are tensors on device
@jax.jit
def solve(
    beam_scores: jax.Array,
    token_logprobs: jax.Array,
    B: int,
    K: int,
    V: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return output tensor directly
    pass
