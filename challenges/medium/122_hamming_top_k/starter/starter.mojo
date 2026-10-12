from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# codes, query, output are device pointers
@export
def solve(
    codes: UnsafePointer[Int32, MutAnyOrigin],
    query: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    k: Int32,
) raises:
    pass
