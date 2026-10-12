import cutlass
import cutlass.cute as cute


# u, delta, A, B, C, skip, y are tensors on the GPU
@cute.jit
def solve(
    u: cute.Tensor,
    delta: cute.Tensor,
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    skip: cute.Tensor,
    y: cute.Tensor,
    batch: cute.Uint32,
    seq_len: cute.Uint32,
    d_model: cute.Uint32,
    d_state: cute.Uint32,
):
    pass
