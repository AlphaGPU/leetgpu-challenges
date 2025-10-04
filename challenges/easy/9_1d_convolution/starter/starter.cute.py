import cutlass
import cutlass.cute as cute

# input, kernel, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, kernel: cute.Tensor, output: cute.Tensor, input_size: cute.Int32, kernel_size: cute.Int32):
    pass
