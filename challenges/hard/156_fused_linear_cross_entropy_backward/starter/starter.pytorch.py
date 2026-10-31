import torch


# X, W, targets, dX, dW are tensors on the GPU
def solve(
    X: torch.Tensor,
    W: torch.Tensor,
    targets: torch.Tensor,
    dX: torch.Tensor,
    dW: torch.Tensor,
    N: int,
    D: int,
    V: int,
):
    pass
