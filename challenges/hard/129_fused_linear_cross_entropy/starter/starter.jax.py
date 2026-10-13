import jax
import jax.numpy as jnp


# X, W, targets are tensors on device
@jax.jit
def solve(X: jax.Array, W: jax.Array, targets: jax.Array, N: int, D: int, V: int) -> jax.Array:
    # return output tensor directly
    pass
