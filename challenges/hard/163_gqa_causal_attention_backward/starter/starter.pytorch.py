import torch


# Q, K, V, O, dO, lse, dQ, dK, dV are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    lse: torch.Tensor,
    dQ: torch.Tensor,
    dK: torch.Tensor,
    dV: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    pass
