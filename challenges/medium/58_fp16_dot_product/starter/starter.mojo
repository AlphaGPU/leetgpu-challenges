from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# A, B, result are device pointers
@export
def solve(
    A: UnsafePointer[Float16, MutAnyOrigin],
    B: UnsafePointer[Float16, MutAnyOrigin],
    result: UnsafePointer[Float16, MutAnyOrigin],
    N: Int32,
) raises:
    pass
