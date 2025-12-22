import jax
import jax.numpy as jnp

# grid is a tensor on the GPU
@jax.jit
def solve(grid: jax.Array, rows: int, cols: int, 
          start_row: int, start_col: int, end_row: int, end_col: int) -> jax.Array:
    # return output tensor directly
    pass
