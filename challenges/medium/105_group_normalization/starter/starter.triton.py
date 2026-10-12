import torch
import triton
import triton.language as tl


# X, gamma, beta, Y are tensors on the GPU
def solve(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    Y: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    G: int,
    eps: float,
):
    pass
