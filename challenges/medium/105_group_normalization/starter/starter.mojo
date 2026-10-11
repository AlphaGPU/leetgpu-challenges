from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# X, gamma, beta, Y are device pointers
@export
def solve(
    X: UnsafePointer[Float32, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    Y: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
    H: Int32,
    W: Int32,
    G: Int32,
    eps: Float32,
) raises:
    pass
