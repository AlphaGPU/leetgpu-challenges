from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K, V, dO, dQ, dK, dV are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    dO: UnsafePointer[Float32, MutAnyOrigin],
    dQ: UnsafePointer[Float32, MutAnyOrigin],
    dK: UnsafePointer[Float32, MutAnyOrigin],
    dV: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    d: Int32,
) raises:
    pass
