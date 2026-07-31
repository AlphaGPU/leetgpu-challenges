import torch


# Q, K, V, q_weight, k_weight, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
    eps: float,
):
    pass
