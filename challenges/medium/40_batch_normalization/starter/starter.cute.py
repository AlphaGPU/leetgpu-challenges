import cutlass
import cutlass.cute as cute

# input, gamma, beta, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, gamma: cute.Tensor, beta: cute.Tensor, output: cute.Tensor, N: cute.Int32, C: cute.Int32, eps: cute.Float32):
    pass
