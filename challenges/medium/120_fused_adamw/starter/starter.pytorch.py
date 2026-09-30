import torch


# params, grad, m, v are tensors on the GPU
def solve(
    params: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    N: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    t: int,
):
    pass
