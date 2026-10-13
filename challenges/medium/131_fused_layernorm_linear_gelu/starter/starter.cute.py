import cutlass
import cutlass.cute as cute


# x, gamma, beta, W, bias, y are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    W: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
    eps: cute.Float32,
):
    pass
