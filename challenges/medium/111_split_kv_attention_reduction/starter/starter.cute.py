import cutlass
import cutlass.cute as cute


# partial_out, partial_lse, output are tensors on the GPU
@cute.jit
def solve(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    num_splits: cute.Int32,
    num_heads: cute.Int32,
    head_dim: cute.Int32,
):
    pass
