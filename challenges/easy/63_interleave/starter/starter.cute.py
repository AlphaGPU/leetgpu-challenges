import cutlass
import cutlass.cute as cute


# A, B, output are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    pass
