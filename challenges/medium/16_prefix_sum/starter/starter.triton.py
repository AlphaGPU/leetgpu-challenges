import torch
import triton
import triton.language as tl

# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    pass
