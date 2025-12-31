import jax
import jax.numpy as jnp

#Q, cos, sin are tensors on the GPU
@jax.jit
def solve(Q: array, cos: array, sin: array, output: array, M: int, D: int):
    # return output tensor directly
    pass
