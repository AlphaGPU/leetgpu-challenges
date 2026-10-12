from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    A: UnsafePointer[Float16, MutAnyOrigin],
    B: UnsafePointer[Float16, MutAnyOrigin],
    C: UnsafePointer[Float16, MutAnyOrigin],
    M: Int32,
    N: Int32,
    K: Int32,
    alpha: Float32,
    beta: Float32,
) raises:
    pass
