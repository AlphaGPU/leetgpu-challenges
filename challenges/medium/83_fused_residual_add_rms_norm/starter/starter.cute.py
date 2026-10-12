import cutlass
import cutlass.cute as cute


# x, residual, weight, out are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    N: cute.Int32,
    C: cute.Int32,
    eps: cute.Float32,
):
    pass
