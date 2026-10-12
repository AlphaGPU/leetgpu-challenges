from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    A: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    nnz: Int32,
) raises:
    pass
