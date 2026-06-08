from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# X, scale, shift, output are device pointers
@export
def solve(
    X: UnsafePointer[Float32, MutExternalOrigin],
    scale: UnsafePointer[Float32, MutExternalOrigin],
    shift: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    B: Int32,
    N: Int32,
    D: Int32,
) raises:
    pass
