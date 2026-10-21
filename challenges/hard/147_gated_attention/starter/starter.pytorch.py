import torch


# Q, K, V, X, W_g, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    X: torch.Tensor,
    W_g: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    h: int,
):
    pass
