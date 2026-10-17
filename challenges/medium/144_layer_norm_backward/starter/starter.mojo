from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    mean: UnsafePointer[Float32, MutAnyOrigin],
    rstd: UnsafePointer[Float32, MutAnyOrigin],
    dy: UnsafePointer[Float32, MutAnyOrigin],
    dx: UnsafePointer[Float32, MutAnyOrigin],
    dgamma: UnsafePointer[Float32, MutAnyOrigin],
    dbeta: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
) raises:
    pass
