import torch
import triton
import triton.language as tl


# A, B, group_offsets, C are tensors on the GPU
def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    group_offsets: torch.Tensor,
    C: torch.Tensor,
    G: int,
    M_total: int,
    K: int,
    N: int,
):
    pass
