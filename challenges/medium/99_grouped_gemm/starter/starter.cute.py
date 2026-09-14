import cutlass
import cutlass.cute as cute


# A, B, group_offsets, C are tensors on the GPU
@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    group_offsets: cute.Tensor,
    C: cute.Tensor,
    G: cute.Int32,
    M_total: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    pass
