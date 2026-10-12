import cutlass
import cutlass.cute as cute


# codes, query, output are tensors on the GPU
@cute.jit
def solve(
    codes: cute.Tensor, query: cute.Tensor, output: cute.Tensor, N: cute.Int32, k: cute.Int32
):
    pass
