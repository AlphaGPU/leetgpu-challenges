import torch
import triton
import triton.language as tl


# src, index, output are tensors on the GPU
def solve(src: torch.Tensor, index: torch.Tensor, output: torch.Tensor, N: int, D: int, M: int):
    pass
