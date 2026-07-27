import torch
import triton
import triton.language as tl


# rewards, values, advantages are tensors on the GPU
def solve(
    rewards: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    lam: float,
    B: int,
    S: int,
):
    pass
