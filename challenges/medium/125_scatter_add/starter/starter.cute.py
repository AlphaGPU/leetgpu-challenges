import cutlass
import cutlass.cute as cute


# src, index, output are tensors on the GPU
@cute.jit
def solve(
    src: cute.Tensor,
    index: cute.Tensor,
    output: cute.Tensor,
    N: cute.Int32,
    D: cute.Int32,
    M: cute.Int32,
):
    pass
