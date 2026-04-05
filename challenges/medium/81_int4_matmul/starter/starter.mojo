from std.gpu.host import DeviceContext
from std.memory import UnsafePointer

# x, w_q, scales, y are device pointers
@export
fn solve(
    x: UnsafePointer[Float16, MutExternalOrigin],
    w_q: UnsafePointer[UInt8, MutExternalOrigin],
    scales: UnsafePointer[Float16, MutExternalOrigin],
    y: UnsafePointer[Float16, MutExternalOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
    group_size: Int32,
) raises:
    pass
