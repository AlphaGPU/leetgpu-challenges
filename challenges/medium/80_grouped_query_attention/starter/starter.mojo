from std.gpu.host import DeviceContext
from std.memory import UnsafePointer

# Q, K, V, output are device pointers
@export
fn solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    num_q_heads: Int32,
    num_kv_heads: Int32,
    seq_len: Int32,
    head_dim: Int32,
) raises:
    pass
