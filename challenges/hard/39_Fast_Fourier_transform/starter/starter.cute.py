import cutlass
import cutlass.cute as cute

# signal, spectrum are tensors on the GPU
@cute.jit
def solve(signal: cute.Tensor, spectrum: cute.Tensor, N: cute.Int32):
    pass
