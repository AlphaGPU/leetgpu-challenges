import cutlass
import cutlass.cute as cute


# G, Buf, output are tensors on the GPU
@cute.jit
def solve(
    G: cute.Tensor,
    Buf: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    momentum: cute.Float32,
):
    pass
