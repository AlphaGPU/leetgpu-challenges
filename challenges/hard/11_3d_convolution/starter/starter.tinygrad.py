import tinygrad

# input, kernel, output are tensors on the GPU
def solve(input: tinygrad.Tensor, kernel: tinygrad.Tensor, output: tinygrad.Tensor,
          input_depth: int, input_rows: int, input_cols: int,
          kernel_depth: int, kernel_rows: int, kernel_cols: int):
    pass 