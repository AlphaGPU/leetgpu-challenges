import cutlass
import cutlass.cute as cute


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, H: cute.Int32, W: cute.Int32):
    pass
