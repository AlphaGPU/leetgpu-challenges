import jax
import jax.numpy as jnp


# value, spatial_shapes, sampling_loc, attn_weight are tensors on device
@jax.jit
def solve(
    value: jax.Array,
    spatial_shapes: jax.Array,
    sampling_loc: jax.Array,
    attn_weight: jax.Array,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    num_levels: int,
    num_points: int,
) -> jax.Array:
    # return output tensor directly
    pass
