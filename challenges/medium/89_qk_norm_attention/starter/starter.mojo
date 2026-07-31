from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K, V, q_weight, k_weight, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    q_weight: UnsafePointer[Float32, MutExternalOrigin],
    k_weight: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    N: Int32,
    d_model: Int32,
    h: Int32,
    eps: Float32,
) raises:
    pass
