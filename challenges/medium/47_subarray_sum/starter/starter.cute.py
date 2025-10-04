import cutlass
import cutlass.cute as cute

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32, S: cute.Int32, E: cute.Int32):
    pass
