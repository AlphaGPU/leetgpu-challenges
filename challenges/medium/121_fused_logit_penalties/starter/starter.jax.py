import jax
import jax.numpy as jnp


# logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty
# are tensors on device
@jax.jit
def solve(
    logits: jax.Array,
    prompt_tokens: jax.Array,
    output_tokens: jax.Array,
    presence_penalty: jax.Array,
    frequency_penalty: jax.Array,
    repetition_penalty: jax.Array,
    B: int,
    V: int,
    P: int,
    G: int,
) -> jax.Array:
    # return output tensor directly
    pass
