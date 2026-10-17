import torch
import triton
import triton.language as tl


# x, gamma, mean, rstd, dy, dx, dgamma, dbeta are tensors on the GPU
def solve(
    x: torch.Tensor,
    gamma: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    dx: torch.Tensor,
    dgamma: torch.Tensor,
    dbeta: torch.Tensor,
    N: int,
    C: int,
):
    pass
