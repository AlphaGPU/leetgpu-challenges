import torch
import triton
import triton.language as tl


# X, dY, group_offsets, dW are tensors on the GPU
def solve(
    X: torch.Tensor,
    dY: torch.Tensor,
    group_offsets: torch.Tensor,
    dW: torch.Tensor,
    G: int,
    M_total: int,
    K: int,
    N: int,
):
    pass
