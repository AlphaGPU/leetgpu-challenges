from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# x, w_q, scales, y are device pointers
@export
def solve(
    x: UnsafePointer[Float16, MutAnyOrigin],
    w_q: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Float16, MutAnyOrigin],
    y: UnsafePointer[Float16, MutAnyOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
    group_size: Int32,
) raises:
    pass
