import cutlass
import cutlass.cute as cute


# X, gamma, beta, Y are tensors on the GPU
@cute.jit
def solve(
    X: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    Y: cute.Tensor,
    N: cute.Int32,
    C: cute.Int32,
    H: cute.Int32,
    W: cute.Int32,
    G: cute.Int32,
    eps: cute.Float32,
):
    pass
