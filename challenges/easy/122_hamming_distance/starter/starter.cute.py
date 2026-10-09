import cutlass
import cutlass.cute as cute


# codes, output are tensors on the GPU
@cute.jit
def solve(codes: cute.Tensor, output: cute.Tensor, query: cute.Int32, N: cute.Int32):
    pass
