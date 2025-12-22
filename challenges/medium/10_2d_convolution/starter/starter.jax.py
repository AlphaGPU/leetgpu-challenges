import jax
import jax.numpy as jnp

# input, kernel are tensors on the GPU
@jax.jit
def solve(input: jax.Array, kernel: jax.Array,
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    pass
