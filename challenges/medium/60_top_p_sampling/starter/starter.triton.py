import torch
import triton
import triton.language as tl


def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    top_p_probs: torch.Tensor,
    vocab_size: int,
):
    pass
