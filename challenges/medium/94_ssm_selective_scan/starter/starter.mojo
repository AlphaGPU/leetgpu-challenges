from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# u, delta, A, B, C, skip, y are device pointers
@export
def solve(
    u: UnsafePointer[Float32, MutAnyOrigin],
    delta: UnsafePointer[Float32, MutAnyOrigin],
    A: UnsafePointer[Float32, MutAnyOrigin],
    B: UnsafePointer[Float32, MutAnyOrigin],
    C: UnsafePointer[Float32, MutAnyOrigin],
    skip: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    batch: Int32,
    seq_len: Int32,
    d_model: Int32,
    d_state: Int32,
) raises:
    pass
