import cutlass
import cutlass.cute as cute


# a, x, h are tensors on the GPU
@cute.jit
def solve(a: cute.Tensor, x: cute.Tensor, h: cute.Tensor, B: cute.Int32, L: cute.Int32):
    pass
