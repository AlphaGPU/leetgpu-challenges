import cutlass
import cutlass.cute as cute


# Q, K_int8, V_int8, k_scale, v_scale, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K_int8: cute.Tensor,
    V_int8: cute.Tensor,
    k_scale: cute.Tensor,
    v_scale: cute.Tensor,
    output: cute.Tensor,
    num_heads: cute.Int32,
    seq_len: cute.Int32,
    head_dim: cute.Int32,
):
    pass
