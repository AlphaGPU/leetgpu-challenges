import torch


# Q, K, V, alpha, beta, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    seq_len: int,
    d: int,
):
    pass
