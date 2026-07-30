from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, residual, weight, out are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    residual: UnsafePointer[Float32, MutExternalOrigin],
    weight: UnsafePointer[Float32, MutExternalOrigin],
    out: UnsafePointer[Float32, MutExternalOrigin],
    N: Int32,
    C: Int32,
    eps: Float32,
) raises:
    pass
