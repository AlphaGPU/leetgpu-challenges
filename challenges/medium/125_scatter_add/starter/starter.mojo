from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# src, index, output are device pointers
@export
def solve(
    src: UnsafePointer[Float32, MutAnyOrigin],
    index: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    D: Int32,
    M: Int32,
) raises:
    pass
