import torch
import triton
import triton.language as tl


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, M: int, N: int):
    pass
