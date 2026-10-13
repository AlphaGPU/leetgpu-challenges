import jax
import jax.numpy as jnp


# x, topk_ids, topk_weights, w_gate, w_up, w_down are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    T: int,
    H: int,
    I: int,
    E: int,
    K: int,
) -> jax.Array:
    # return output tensor directly
    pass
