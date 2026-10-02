import torch


# partial_out, partial_lse, output are tensors on the GPU
def solve(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    num_splits: int,
    num_heads: int,
    head_dim: int,
):
    pass
