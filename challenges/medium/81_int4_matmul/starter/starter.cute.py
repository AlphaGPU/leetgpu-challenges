import cutlass
import cutlass.cute as cute


# x, scales, y are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    w_q: cute.Tensor,
    scales: cute.Tensor,
    y: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
    group_size: cute.Int32,
):
    pass
