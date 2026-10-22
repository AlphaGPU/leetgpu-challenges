import cutlass
import cutlass.cute as cute


# x, gamma, W, w_scale, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    gamma: cute.Tensor,
    W: cute.Tensor,
    w_scale: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    pass
