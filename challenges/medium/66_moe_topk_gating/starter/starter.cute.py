import cutlass
import cutlass.cute as cute


# logits, topk_weights, topk_indices are tensors on the GPU
@cute.jit
def solve(
    logits: cute.Tensor,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    M: cute.Int32,
    E: cute.Int32,
    k: cute.Int32,
):
    pass
