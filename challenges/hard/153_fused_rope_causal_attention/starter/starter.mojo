from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K, V, cos, sin, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    cos: UnsafePointer[Float32, MutAnyOrigin],
    sin: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    num_q_heads: Int32,
    num_kv_heads: Int32,
    seq_len: Int32,
    head_dim: Int32,
) raises:
    pass
