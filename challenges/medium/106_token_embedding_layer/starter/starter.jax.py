import jax
import jax.numpy as jnp


# token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta are tensors on device
@jax.jit
def solve(
    token_ids: jax.Array,
    position_ids: jax.Array,
    token_embeddings: jax.Array,
    position_embeddings: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    B: int,
    T: int,
    V: int,
    P: int,
    D: int,
    eps: float,
) -> jax.Array:
    # return output tensor directly
    pass
