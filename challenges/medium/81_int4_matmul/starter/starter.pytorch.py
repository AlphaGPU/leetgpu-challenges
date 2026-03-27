import torch


# x, w_q, scales, y are tensors on the GPU
def solve(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    pass
