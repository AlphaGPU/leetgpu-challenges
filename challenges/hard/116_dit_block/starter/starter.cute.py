import cutlass
import cutlass.cute as cute


# x, c, output, weights are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    c: cute.Tensor,
    output: cute.Tensor,
    weights: cute.Tensor,
    batch_size: cute.Int32,
    seq_len: cute.Int32,
):
    pass
