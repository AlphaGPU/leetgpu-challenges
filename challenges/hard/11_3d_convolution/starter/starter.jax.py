import jax
import jax.numpy as jnp

# input, kernel are tensors on the GPU
@jax.jit
def solve(input: jax.Array, kernel: jax.Array,
          input_depth: int, input_rows: int, input_cols: int,
          kernel_depth: int, kernel_rows: int, kernel_cols: int):
    pass
