import cutlass
import cutlass.cute as cute


# X, W, targets, dX, dW are tensors on the GPU
@cute.jit
def solve(
    X: cute.Tensor,
    W: cute.Tensor,
    targets: cute.Tensor,
    dX: cute.Tensor,
    dW: cute.Tensor,
    N: cute.Int32,
    D: cute.Int32,
    V: cute.Int32,
):
    pass
