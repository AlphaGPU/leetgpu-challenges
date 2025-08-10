import torch
import triton
import triton.language as tl

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    pass 