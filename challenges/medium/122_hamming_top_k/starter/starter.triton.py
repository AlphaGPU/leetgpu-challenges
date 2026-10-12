import torch
import triton
import triton.language as tl


# codes, query, output are tensors on the GPU
def solve(codes: torch.Tensor, query: torch.Tensor, output: torch.Tensor, N: int, k: int):
    pass
