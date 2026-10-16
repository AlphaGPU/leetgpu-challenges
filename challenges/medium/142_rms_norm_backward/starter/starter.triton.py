import torch
import triton
import triton.language as tl


# x, weight, dy, dx, dweight are tensors on the GPU
def solve(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    pass
