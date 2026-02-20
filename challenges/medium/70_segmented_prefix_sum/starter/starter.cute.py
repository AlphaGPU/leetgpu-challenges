import cutlass
import cutlass.cute as cute


# values, flags, output are tensors on the GPU
@cute.jit
def solve(values: cute.Tensor, flags: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    pass
