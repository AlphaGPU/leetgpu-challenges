import cutlass
import cutlass.cute as cute


# logits, input_ids are tensors on the GPU
@cute.jit
def solve(
    logits: cute.Tensor,
    input_ids: cute.Tensor,
    penalty: cute.Float32,
    B: cute.Int32,
    V: cute.Int32,
    T: cute.Int32,
):
    pass
