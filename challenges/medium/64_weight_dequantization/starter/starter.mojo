from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# X, S, Y are device pointers
@export
def solve(
    X: UnsafePointer[Float32, MutAnyOrigin],
    S: UnsafePointer[Float32, MutAnyOrigin],
    Y: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
    TILE_SIZE: Int32,
) raises:
    pass
