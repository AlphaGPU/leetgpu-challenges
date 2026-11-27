import cutlass
import cutlass.cute as cute


# Q, K, scores are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    scores: cute.Tensor,
    num_q_heads: cute.Int32,
    num_kv_heads: cute.Int32,
    window: cute.Int32,
    seq_len: cute.Int32,
    head_dim: cute.Int32,
):
    pass
