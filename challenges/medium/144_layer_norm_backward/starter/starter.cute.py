import cutlass
import cutlass.cute as cute


# x, gamma, mean, rstd, dy, dx, dgamma, dbeta are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    gamma: cute.Tensor,
    mean: cute.Tensor,
    rstd: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dgamma: cute.Tensor,
    dbeta: cute.Tensor,
    N: cute.Int32,
    C: cute.Int32,
):
    pass
