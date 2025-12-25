import jax
import jax.numpy as jnp


# X, y are tensors on the GPU
@jax.jit
def solve(X: jax.Array, y: jax.Array, n_samples: int, n_features: int) -> jax.Array:
    # return output tensor directly
    pass
