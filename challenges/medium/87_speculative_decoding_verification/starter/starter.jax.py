import jax
import jax.numpy as jnp


# draft_tokens, draft_probs, target_probs, uniform_samples are tensors on device
@jax.jit
def solve(
    draft_tokens: jax.Array,
    draft_probs: jax.Array,
    target_probs: jax.Array,
    uniform_samples: jax.Array,
    B: int,
    T: int,
    V: int,
) -> jax.Array:
    # return output tensor directly
    pass
