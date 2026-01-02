import cutlass
import cutlass.cute as cute


# Q, cos, sin, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    D: cute.Int32,
):
    pass
