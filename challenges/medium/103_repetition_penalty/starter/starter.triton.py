import torch
import triton
import triton.language as tl


# logits, input_ids are tensors on the GPU
def solve(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
    B: int,
    V: int,
    T: int,
):
    pass
