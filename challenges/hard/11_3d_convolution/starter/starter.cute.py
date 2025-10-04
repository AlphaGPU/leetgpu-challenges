import cutlass
import cutlass.cute as cute

# input, kernel, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, kernel: cute.Tensor, output: cute.Tensor, input_depth: cute.Int32, input_rows: cute.Int32, input_cols: cute.Int32, kernel_depth: cute.Int32, kernel_rows: cute.Int32, kernel_cols: cute.Int32):
    pass
