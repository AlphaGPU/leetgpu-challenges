import cutlass
import cutlass.cute as cute

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, gamma: cute.Float32, beta: cute.Float32, output: cute.Tensor, N: cute.Int32, eps: cute.Float32):
    pass
