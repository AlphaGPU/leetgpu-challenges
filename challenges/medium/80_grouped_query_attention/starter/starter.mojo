from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    num_q_heads: Int32,
    num_kv_heads: Int32,
    seq_len: Int32,
    head_dim: Int32,
) raises:
    pass
