import torch
import triton
import triton.language as tl


# input, gamma, beta, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
):
    pass
