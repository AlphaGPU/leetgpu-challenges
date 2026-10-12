import torch


# u, delta, A, B, C, skip, y are tensors on the GPU
def solve(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    skip: torch.Tensor,
    y: torch.Tensor,
    batch: int,
    seq_len: int,
    d_model: int,
    d_state: int,
):
    pass
