from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    residual: UnsafePointer[Float32, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    residual_out: UnsafePointer[Float32, MutAnyOrigin],
    mean: UnsafePointer[Float32, MutAnyOrigin],
    rstd: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
    eps: Float32,
) raises:
    pass
