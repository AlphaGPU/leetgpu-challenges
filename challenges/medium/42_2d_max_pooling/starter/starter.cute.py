import cutlass
import cutlass.cute as cute

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32, C: cute.Int32, H: cute.Int32, W: cute.Int32, kernel_size: cute.Int32, stride: cute.Int32, padding: cute.Int32):
    pass
