import cutlass
import cutlass.cute as cute


# logits, output are tensors on the GPU
@cute.jit
def solve(logits: cute.Tensor, output: cute.Tensor, B: cute.Int32, V: cute.Int32):
    pass
