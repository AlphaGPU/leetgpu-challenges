from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int32,
    d_model: Int32,
    gamma: Float32,
) raises:
    pass
