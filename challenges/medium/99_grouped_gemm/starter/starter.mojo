from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# A, B, group_offsets, C are device pointers
@export
def solve(
    A: UnsafePointer[Float32, MutExternalOrigin],
    B: UnsafePointer[Float32, MutExternalOrigin],
    group_offsets: UnsafePointer[Int32, MutExternalOrigin],
    C: UnsafePointer[Float32, MutExternalOrigin],
    G: Int32,
    M_total: Int32,
    K: Int32,
    N: Int32,
) raises:
    pass
