import torch
import triton
import triton.language as tl


# dist, output are tensors on the GPU (N*N floats each, row-major)
def solve(dist: torch.Tensor, output: torch.Tensor, N: int):
    pass
