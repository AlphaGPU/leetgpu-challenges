import torch
import triton
import triton.language as tl


# X, W, targets, loss are tensors on the GPU
def solve(
    X: torch.Tensor,
    W: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    N: int,
    D: int,
    V: int,
):
    pass
