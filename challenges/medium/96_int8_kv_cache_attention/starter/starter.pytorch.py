import torch


# Q, K_int8, V_int8, k_scale, v_scale, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    pass
