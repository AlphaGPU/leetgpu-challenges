import cutlass
import cutlass.cute as cute


# Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K_cache: cute.Tensor,
    V_cache: cute.Tensor,
    block_table: cute.Tensor,
    context_lens: cute.Tensor,
    cu_seqlens_q: cute.Tensor,
    output: cute.Tensor,
    num_seqs: cute.Int32,
    total_q: cute.Int32,
    num_q_heads: cute.Int32,
    num_kv_heads: cute.Int32,
    head_dim: cute.Int32,
    block_size: cute.Int32,
    max_blocks_per_seq: cute.Int32,
):
    pass
