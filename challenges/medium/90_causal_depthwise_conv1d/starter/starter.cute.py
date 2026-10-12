import cutlass
import cutlass.cute as cute


# x, weight, bias, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
    B: cute.Int32,
    L: cute.Int32,
    D: cute.Int32,
    K: cute.Int32,
):
    pass
