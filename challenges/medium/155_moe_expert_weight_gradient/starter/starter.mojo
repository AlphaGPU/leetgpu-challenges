from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# X, dY, group_offsets, dW are device pointers
@export
def solve(
    X: UnsafePointer[Float32, MutAnyOrigin],
    dY: UnsafePointer[Float32, MutAnyOrigin],
    group_offsets: UnsafePointer[Int32, MutAnyOrigin],
    dW: UnsafePointer[Float32, MutAnyOrigin],
    G: Int32,
    M_total: Int32,
    K: Int32,
    N: Int32,
) raises:
    pass
