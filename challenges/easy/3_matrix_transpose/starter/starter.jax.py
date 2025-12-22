import jax
import jax.numpy as jnp

#input is a tensor on GPU
@jax.jit
def solve(input: jax.Array, rows: int, cols: int):
    pass