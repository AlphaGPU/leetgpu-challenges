import torch


# x, dt, dt_bias, A, B, C, D, z, y are tensors on the GPU
def solve(
    x: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_groups: int,
    state_dim: int,
):
    pass
