import cutlass
import cutlass.cute as cute


# dist, output are tensors on the GPU (N*N floats each, row-major)
@cute.jit
def solve(dist: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    pass
