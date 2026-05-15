import cutlass
import cutlass.cute as cute


# X, scale, shift, output are tensors on the GPU
@cute.jit
def solve(
    X: cute.Tensor,
    scale: cute.Tensor,
    shift: cute.Tensor,
    output: cute.Tensor,
    B: cute.Int32,
    N: cute.Int32,
    D: cute.Int32,
):
    pass
