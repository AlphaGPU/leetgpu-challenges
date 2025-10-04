import cutlass
import cutlass.cute as cute

# input are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, N: cute.Int32):
    pass
