import torch
import triton
import triton.language as tl


# logits, output are tensors on the GPU
def solve(logits: torch.Tensor, output: torch.Tensor, B: int, V: int):
    pass
