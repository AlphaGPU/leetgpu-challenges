import jax
import jax.numpy as jnp


# images, patch_weight, patch_bias, cls_token, pos_embed are tensors on device
@jax.jit
def solve(
    images: jax.Array,
    patch_weight: jax.Array,
    patch_bias: jax.Array,
    cls_token: jax.Array,
    pos_embed: jax.Array,
    B: int,
    C: int,
    H: int,
    W: int,
    P: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
