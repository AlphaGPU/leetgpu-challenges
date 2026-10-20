import torch


# Q, K, V, lambdas, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lambdas: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    lambda_init: float,
):
    pass
