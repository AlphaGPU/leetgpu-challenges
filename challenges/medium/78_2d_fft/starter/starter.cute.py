import cutlass
import cutlass.cute as cute


# signal, spectrum are tensors on the GPU
@cute.jit
def solve(signal: cute.Tensor, spectrum: cute.Tensor, M: cute.Int32, N: cute.Int32):
    pass
