import jax
import jax.numpy as jnp


# logits, input_ids are tensors on GPU
@jax.jit
def solve(
    logits: jax.Array,
    input_ids: jax.Array,
    penalty: float,
    B: int,
    V: int,
    T: int,
) -> jax.Array:
    # return output tensor directly
    pass
