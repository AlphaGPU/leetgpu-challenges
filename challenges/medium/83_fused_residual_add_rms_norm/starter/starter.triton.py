import torch
import triton
import triton.language as tl


# x, residual, weight, out are tensors on the GPU
def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    pass
