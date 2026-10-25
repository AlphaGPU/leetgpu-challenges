from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# G, Buf, output are device pointers
@export
def solve(
    G: UnsafePointer[Float32, MutAnyOrigin],
    Buf: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    momentum: Float32,
) raises:
    pass
