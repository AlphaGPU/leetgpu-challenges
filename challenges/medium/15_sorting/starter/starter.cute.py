import cutlass
import cutlass.cute as cute

# data are tensors on the GPU
@cute.jit
def solve(data: cute.Tensor, N: cute.Int32):
    pass
