import torch
import triton
import triton.language as tl

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int, alpha: float, beta: float):
    pass
