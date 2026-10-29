import cutlass
import cutlass.cute as cute


# x, dt, dt_bias, A, B, C, D, z, y are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    dt: cute.Tensor,
    dt_bias: cute.Tensor,
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    D: cute.Tensor,
    z: cute.Tensor,
    y: cute.Tensor,
    batch: cute.Int32,
    seq_len: cute.Int32,
    num_heads: cute.Int32,
    head_dim: cute.Int32,
    num_groups: cute.Int32,
    state_dim: cute.Int32,
):
    pass
