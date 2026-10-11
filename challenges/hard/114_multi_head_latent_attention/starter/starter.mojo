from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# q, kv_cache, W_UK, W_UV, output are device pointers
@export
def solve(
    q: UnsafePointer[Float32, MutAnyOrigin],
    kv_cache: UnsafePointer[Float32, MutAnyOrigin],
    W_UK: UnsafePointer[Float32, MutAnyOrigin],
    W_UV: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    num_heads: Int32,
    seq_len: Int32,
    kv_lora_rank: Int32,
    head_dim: Int32,
    rope_dim: Int32,
) raises:
    pass
