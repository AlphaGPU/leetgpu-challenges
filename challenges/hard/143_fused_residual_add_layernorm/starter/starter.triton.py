import torch
import triton
import triton.language as tl


# x, residual, gamma, beta, y, residual_out, mean, rstd are tensors on the GPU
def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    residual_out: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    pass
