import cutlass
import cutlass.cute as cute


# X, Y are tensors on the GPU
@cute.jit
def solve(X: cute.Tensor, Y: cute.Tensor, N: cute.Uint32):
    pass
