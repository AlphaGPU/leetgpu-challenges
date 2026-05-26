import torch


# x, residual, weight, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    eps: float,
):
    pass
