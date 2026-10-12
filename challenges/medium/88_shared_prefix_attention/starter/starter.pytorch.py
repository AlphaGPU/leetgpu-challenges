import torch


# Q, K_prefix, V_prefix, K_suffix, V_suffix, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_prefix: torch.Tensor,
    V_prefix: torch.Tensor,
    K_suffix: torch.Tensor,
    V_suffix: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    num_heads: int,
    prefix_len: int,
    suffix_len: int,
    head_dim: int,
):
    pass
