import torch
import triton
import triton.language as tl


# x, gamma, beta, W, bias, y are tensors on the GPU
def solve(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    M: int,
    K: int,
    N: int,
    eps: float,
):
    pass
