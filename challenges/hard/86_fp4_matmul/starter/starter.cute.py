import cutlass
import cutlass.cute as cute


# x_q, x_scales, w_q, w_scales, y are tensors on the GPU
@cute.jit
def solve(
    x_q: cute.Tensor,
    x_scales: cute.Tensor,
    w_q: cute.Tensor,
    w_scales: cute.Tensor,
    alpha: cute.Float32,
    y: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    pass
