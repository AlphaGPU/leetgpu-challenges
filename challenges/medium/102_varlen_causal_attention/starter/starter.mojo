from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, cu_seqlens, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    cu_seqlens: UnsafePointer[Int32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    T: Int32,
    d: Int32,
    S: Int32,
) raises:
    pass
