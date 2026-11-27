import torch
import triton
import triton.language as tl


# Q, K, scores are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    scores: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    window: int,
    seq_len: int,
    head_dim: int,
):
    pass
