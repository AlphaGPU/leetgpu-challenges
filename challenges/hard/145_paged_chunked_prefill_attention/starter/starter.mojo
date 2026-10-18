from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K_cache: UnsafePointer[Float32, MutAnyOrigin],
    V_cache: UnsafePointer[Float32, MutAnyOrigin],
    block_table: UnsafePointer[Int32, MutAnyOrigin],
    context_lens: UnsafePointer[Int32, MutAnyOrigin],
    cu_seqlens_q: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    num_seqs: Int32,
    total_q: Int32,
    num_q_heads: Int32,
    num_kv_heads: Int32,
    head_dim: Int32,
    block_size: Int32,
    max_blocks_per_seq: Int32,
) raises:
    pass
