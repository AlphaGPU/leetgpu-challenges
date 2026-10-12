import jax
import jax.numpy as jnp


# y_samples is a tensor on device
@jax.jit
def solve(y_samples: jax.Array, a: float, b: float, n_samples: int) -> jax.Array:
    # return output tensor directly
    pass
