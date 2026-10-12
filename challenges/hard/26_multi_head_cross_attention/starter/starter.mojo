from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    H: Int32,
    D: Int32,
) raises:
    pass
