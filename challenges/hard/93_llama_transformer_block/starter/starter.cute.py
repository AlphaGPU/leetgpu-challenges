import cutlass
import cutlass.cute as cute


# x, output, weights, cos, sin are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    output: cute.Tensor,
    weights: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    seq_len: cute.Int32,
):
    pass
