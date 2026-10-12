from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, alpha, beta, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    alpha: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch: Int32,
    seq_len: Int32,
    d: Int32,
) raises:
    pass
