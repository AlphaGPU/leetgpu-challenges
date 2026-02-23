import torch
import triton
import triton.language as tl


# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    pass
