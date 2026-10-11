from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, cu_seqlens, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    cu_seqlens: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    T: Int32,
    d: Int32,
    S: Int32,
) raises:
    pass
