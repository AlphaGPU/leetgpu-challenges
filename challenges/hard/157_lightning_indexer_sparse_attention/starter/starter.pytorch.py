import torch


# q, k_cache, v_cache, idx_q, idx_k, idx_w, output are tensors on the GPU
def solve(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    output: torch.Tensor,
    B: int,
    S: int,
    H: int,
    D: int,
    H_I: int,
    D_I: int,
    topk: int,
):
    pass
