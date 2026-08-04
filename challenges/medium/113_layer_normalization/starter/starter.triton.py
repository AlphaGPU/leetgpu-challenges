import torch
import triton
import triton.language as tl


# input, weight, bias, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    pass
