from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# u, delta, A, B, C, skip, y are device pointers
@export
def solve(
    u: UnsafePointer[Float32, MutExternalOrigin],
    delta: UnsafePointer[Float32, MutExternalOrigin],
    A: UnsafePointer[Float32, MutExternalOrigin],
    B: UnsafePointer[Float32, MutExternalOrigin],
    C: UnsafePointer[Float32, MutExternalOrigin],
    skip: UnsafePointer[Float32, MutExternalOrigin],
    y: UnsafePointer[Float32, MutExternalOrigin],
    batch: Int32,
    seq_len: Int32,
    d_model: Int32,
    d_state: Int32,
) raises:
    pass
