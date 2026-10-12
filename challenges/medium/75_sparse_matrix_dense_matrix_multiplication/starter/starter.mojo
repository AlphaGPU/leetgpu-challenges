from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# A, B, C are device pointers
@export
def solve(
    A: UnsafePointer[Float32, MutAnyOrigin],
    B: UnsafePointer[Float32, MutAnyOrigin],
    C: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
    nnz: Int32,
) raises:
    pass
