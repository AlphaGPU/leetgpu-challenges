import cutlass
import cutlass.cute as cute


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor, output: cute.Tensor, lo: cute.Float32, hi: cute.Float32, N: cute.Int32
):
    pass
