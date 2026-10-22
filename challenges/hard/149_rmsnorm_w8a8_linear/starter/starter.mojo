from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, gamma, W, w_scale, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    W: UnsafePointer[Int8, MutAnyOrigin],
    w_scale: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
) raises:
    pass
