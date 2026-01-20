import cutlass
import cutlass.cute as cute


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, width: cute.Uint32, height: cute.Uint32):
    pass