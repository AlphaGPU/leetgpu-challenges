import jax
import jax.numpy as jnp


# params, grad, m, v are tensors on device
@jax.jit
def solve(
    params: jax.Array,
    grad: jax.Array,
    m: jax.Array,
    v: jax.Array,
    N: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    t: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # return (params, m, v) tensors directly
    pass
