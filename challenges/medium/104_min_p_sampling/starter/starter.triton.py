import torch
import triton
import triton.language as tl


# logits, probs are tensors on the GPU
def solve(logits: torch.Tensor, probs: torch.Tensor, min_p: float, B: int, V: int):
    pass
