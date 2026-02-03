import torch
import triton
import triton.language as tl


# logits, topk_weights, topk_indices are tensors on the GPU
def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    pass
