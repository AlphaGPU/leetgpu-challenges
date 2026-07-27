from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K, V, dO, dQ, dK, dV are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    dO: UnsafePointer[Float32, MutExternalOrigin],
    dQ: UnsafePointer[Float32, MutExternalOrigin],
    dK: UnsafePointer[Float32, MutExternalOrigin],
    dV: UnsafePointer[Float32, MutExternalOrigin],
    M: Int32,
    N: Int32,
    d: Int32,
) raises:
    pass
