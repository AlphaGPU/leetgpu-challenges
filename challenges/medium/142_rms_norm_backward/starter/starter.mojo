from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, weight, dy, dx, dweight are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    dy: UnsafePointer[Float32, MutAnyOrigin],
    dx: UnsafePointer[Float32, MutAnyOrigin],
    dweight: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
    eps: Float32,
) raises:
    pass
