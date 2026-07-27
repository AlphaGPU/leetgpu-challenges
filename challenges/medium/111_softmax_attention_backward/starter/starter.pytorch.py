import torch


# Q, K, V, dO, dQ, dK, dV are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    dQ: torch.Tensor,
    dK: torch.Tensor,
    dV: torch.Tensor,
    M: int,
    N: int,
    d: int,
):
    pass
