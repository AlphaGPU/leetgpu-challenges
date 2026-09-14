import torch
import triton
import triton.language as tl


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    cache_len: int,
    new_len: int,
    head_dim: int,
):
    pass
