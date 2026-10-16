import cutlass
import cutlass.cute as cute


# x, weight, dy, dx, dweight are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    weight: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dweight: cute.Tensor,
    N: cute.Int32,
    C: cute.Int32,
    eps: cute.Float32,
):
    pass
