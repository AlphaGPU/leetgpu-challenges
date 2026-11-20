from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K, V, O, dO, lse, dQ, dK, dV are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    O: UnsafePointer[Float32, MutAnyOrigin],
    dO: UnsafePointer[Float32, MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    dQ: UnsafePointer[Float32, MutAnyOrigin],
    dK: UnsafePointer[Float32, MutAnyOrigin],
    dV: UnsafePointer[Float32, MutAnyOrigin],
    num_q_heads: Int32,
    num_kv_heads: Int32,
    seq_len: Int32,
    head_dim: Int32,
) raises:
    pass
