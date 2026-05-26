from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# x, residual, weight, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    residual: UnsafePointer[Float32, MutExternalOrigin],
    weight: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    M: Int32,
    N: Int32,
    eps: Float32,
) raises:
    pass
