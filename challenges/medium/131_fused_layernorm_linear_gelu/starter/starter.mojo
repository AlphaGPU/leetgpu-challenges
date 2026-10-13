from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, gamma, beta, W, bias, y are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    W: UnsafePointer[Float32, MutAnyOrigin],
    bias: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
    eps: Float32,
) raises:
    pass
