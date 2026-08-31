from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Q, K_cache, V_cache, block_table, context_lens, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32],
    K_cache: UnsafePointer[Float32],
    V_cache: UnsafePointer[Float32],
    block_table: UnsafePointer[Int32],
    context_lens: UnsafePointer[Int32],
    output: UnsafePointer[Float32],
    batch_size: Int32,
    num_heads: Int32,
    head_dim: Int32,
    block_size: Int32,
    max_blocks_per_seq: Int32,
):
    pass
