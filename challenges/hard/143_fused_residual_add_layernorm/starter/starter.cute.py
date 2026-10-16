import cutlass
import cutlass.cute as cute


# x, residual, gamma, beta, y, residual_out, mean, rstd are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    residual: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    y: cute.Tensor,
    residual_out: cute.Tensor,
    mean: cute.Tensor,
    rstd: cute.Tensor,
    N: cute.Int32,
    C: cute.Int32,
    eps: cute.Float32,
):
    pass
