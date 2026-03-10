import cutlass
import cutlass.cute as cute


# Q, K, V, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    output: cute.Tensor,
    num_q_heads: cute.Int32,
    num_kv_heads: cute.Int32,
    seq_len: cute.Int32,
    head_dim: cute.Int32,
):
    pass
