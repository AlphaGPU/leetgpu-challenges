from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# X, gamma, beta, Y are device pointers
@export
def solve(
    X: UnsafePointer[Float32, MutExternalOrigin],
    gamma: UnsafePointer[Float32, MutExternalOrigin],
    beta: UnsafePointer[Float32, MutExternalOrigin],
    Y: UnsafePointer[Float32, MutExternalOrigin],
    N: Int32,
    C: Int32,
    H: Int32,
    W: Int32,
    G: Int32,
    eps: Float32,
) raises:
    pass
