import torch
import triton
import triton.language as tl


# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    pass
