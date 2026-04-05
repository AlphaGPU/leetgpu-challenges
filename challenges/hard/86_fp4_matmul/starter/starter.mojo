from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# x_q, x_scales, w_q, w_scales, y are device pointers
@export
def solve(
    x_q: UnsafePointer[UInt8, MutExternalOrigin],
    x_scales: UnsafePointer[UInt8, MutExternalOrigin],
    w_q: UnsafePointer[UInt8, MutExternalOrigin],
    w_scales: UnsafePointer[UInt8, MutExternalOrigin],
    alpha: Float32,
    y: UnsafePointer[Float16, MutExternalOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
) raises:
    pass
