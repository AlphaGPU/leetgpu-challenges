from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    M: Int32,
    N: Int32,
    H: Int32,
    D: Int32,
) raises:
    pass
