import torch
import triton
import triton.language as tl


# boxes, scores, keep are tensors on the GPU
def solve(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    keep: torch.Tensor,
    N: int,
    iou_threshold: float,
):
    pass
