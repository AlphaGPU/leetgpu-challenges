import torch
import triton
import triton.language as tl


# q, k, v, output are tensors on the GPU
def solve(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    B: int,
    H: int,
    H_KV: int,
    S: int,
    D: int,
):
    pass
