from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K, V, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    seq_len: Int32,
    d_model: Int32,
    gamma: Float32,
) raises:
    pass
