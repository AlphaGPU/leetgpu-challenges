import cutlass
import cutlass.cute as cute


# input, gamma, beta, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
):
    pass
