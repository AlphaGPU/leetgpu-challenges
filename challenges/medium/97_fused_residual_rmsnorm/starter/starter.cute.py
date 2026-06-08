import cutlass
import cutlass.cute as cute


# x, residual, weight, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    eps: cute.Float32,
):
    pass
