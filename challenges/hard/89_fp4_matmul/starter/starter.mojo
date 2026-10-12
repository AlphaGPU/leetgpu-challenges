from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x_q, x_scales, w_q, w_scales, y are device pointers
@export
def solve(
    x_q: UnsafePointer[UInt8, MutAnyOrigin],
    x_scales: UnsafePointer[UInt8, MutAnyOrigin],
    w_q: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales: UnsafePointer[UInt8, MutAnyOrigin],
    alpha: Float32,
    y: UnsafePointer[Float16, MutAnyOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
) raises:
    pass
