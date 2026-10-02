from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, alpha, beta, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    alpha: UnsafePointer[Float32, MutExternalOrigin],
    beta: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    batch: Int32,
    seq_len: Int32,
    d: Int32,
) raises:
    pass
