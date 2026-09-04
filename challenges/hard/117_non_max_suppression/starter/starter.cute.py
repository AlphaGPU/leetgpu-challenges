import cutlass
import cutlass.cute as cute


# boxes, scores, keep are tensors on the GPU
@cute.jit
def solve(
    boxes: cute.Tensor,
    scores: cute.Tensor,
    keep: cute.Tensor,
    N: cute.Int32,
    iou_threshold: cute.Float32,
):
    pass
