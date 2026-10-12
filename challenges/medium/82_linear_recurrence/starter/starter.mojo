from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# a, x, h are device pointers
@export
def solve(
    a: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    h: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    L: Int32,
) raises:
    pass
