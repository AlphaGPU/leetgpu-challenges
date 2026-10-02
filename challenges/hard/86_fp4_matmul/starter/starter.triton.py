import torch
import triton
import triton.language as tl


# x_q, x_scales, w_q, w_scales, y are tensors on the GPU
def solve(
    x_q: torch.Tensor,
    x_scales: torch.Tensor,
    w_q: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: float,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
):
    pass
