import cutlass
import cutlass.cute as cute


# dist, output are tensors on the GPU
@cute.jit
def solve(dist: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    pass
