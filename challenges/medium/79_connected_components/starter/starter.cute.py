import cutlass
import cutlass.cute as cute


# src, dst, labels are tensors on the GPU
@cute.jit
def solve(
    src: cute.Tensor,
    dst: cute.Tensor,
    labels: cute.Tensor,
    N: cute.Int32,
    M: cute.Int32,
):
    pass
