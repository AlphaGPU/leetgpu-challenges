from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# src, dst, labels are device pointers
@export
def solve(
    src: UnsafePointer[Int32, MutAnyOrigin],
    dst: UnsafePointer[Int32, MutAnyOrigin],
    labels: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    M: Int32,
) raises:
    pass
