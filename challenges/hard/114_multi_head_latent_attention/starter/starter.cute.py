import cutlass
import cutlass.cute as cute


# q, kv_cache, W_UK, W_UV, output are tensors on the GPU
@cute.jit
def solve(
    q: cute.Tensor,
    kv_cache: cute.Tensor,
    W_UK: cute.Tensor,
    W_UV: cute.Tensor,
    output: cute.Tensor,
    num_heads: cute.Int32,
    seq_len: cute.Int32,
    kv_lora_rank: cute.Int32,
    head_dim: cute.Int32,
    rope_dim: cute.Int32,
):
    pass
