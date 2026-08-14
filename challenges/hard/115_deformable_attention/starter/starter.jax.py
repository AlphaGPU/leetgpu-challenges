import jax
import jax.numpy as jnp


# value, spatial_shapes, sampling_loc, attn_weight are tensors on device
@jax.jit
def solve(
    value: jax.Array,
    spatial_shapes: jax.Array,
    sampling_loc: jax.Array,
    attn_weight: jax.Array,
    N: int,
    S: int,
    Q: int,
    H: int,
    D: int,
    L: int,
    P: int,
) -> jax.Array:
    # return output tensor directly
    pass
