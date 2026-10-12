from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# values, flags, output are device pointers
@export
def solve(
    values: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
) raises:
    pass
