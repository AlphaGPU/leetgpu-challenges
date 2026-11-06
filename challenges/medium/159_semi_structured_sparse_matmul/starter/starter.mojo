from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, w_values, w_meta, y are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    w_values: UnsafePointer[Float32, MutAnyOrigin],
    w_meta: UnsafePointer[UInt8, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
) raises:
    pass
