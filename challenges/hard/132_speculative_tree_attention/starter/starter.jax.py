import jax
import jax.numpy as jnp


# Q, K_cache, V_cache, K_tree, V_tree, parents are tensors on device
@jax.jit
def solve(
    Q: jax.Array,
    K_cache: jax.Array,
    V_cache: jax.Array,
    K_tree: jax.Array,
    V_tree: jax.Array,
    parents: jax.Array,
    P: int,
    T: int,
    H: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
