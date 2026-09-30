import cutlass
import cutlass.cute as cute


# logits, probs are tensors on the GPU
@cute.jit
def solve(
    logits: cute.Tensor,
    probs: cute.Tensor,
    min_p: cute.Float32,
    B: cute.Int32,
    V: cute.Int32,
):
    pass
