import cutlass
import cutlass.cute as cute


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor, output: cute.Tensor, N: cute.Int32, lo: cute.Float32, hi: cute.Float32
):
    pass
