import cutlass
import cutlass.cute as cute


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, width: cute.Int32, height: cute.Int32):
    pass
