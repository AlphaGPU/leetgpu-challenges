import torch
import triton
import triton.language as tl

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
          input_depth: int, input_rows: int, input_cols: int,
          kernel_depth: int, kernel_rows: int, kernel_cols: int):
    pass 