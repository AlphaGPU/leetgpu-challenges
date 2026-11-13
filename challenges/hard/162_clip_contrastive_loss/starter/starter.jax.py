import jax
import jax.numpy as jnp


# image_embeds, text_embeds are tensors on device
@jax.jit
def solve(
    image_embeds: jax.Array, text_embeds: jax.Array, N: int, D: int, logit_scale: float
) -> jax.Array:
    # return output tensor directly
    pass
