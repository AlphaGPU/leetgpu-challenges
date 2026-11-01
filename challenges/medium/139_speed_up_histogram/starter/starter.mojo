from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# input, histogram are device pointers
@export
def solve(
    input: UnsafePointer[Int32, MutAnyOrigin],
    histogram: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    num_bins: Int32,
) raises:
    pass
