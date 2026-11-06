import cutlass
import cutlass.cute as cute


# x, w_values, w_meta, y are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    w_values: cute.Tensor,
    w_meta: cute.Tensor,
    y: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    pass
