import cutlass
import cutlass.cute as cute


# X, dY, group_offsets, dW are tensors on the GPU
@cute.jit
def solve(
    X: cute.Tensor,
    dY: cute.Tensor,
    group_offsets: cute.Tensor,
    dW: cute.Tensor,
    G: cute.Int32,
    M_total: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    pass
