from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# X, S, Y are device pointers
@export
fn solve(X: UnsafePointer[Float32, MutExternalOrigin], S: UnsafePointer[Float32, MutExternalOrigin], Y: UnsafePointer[Float32, MutExternalOrigin], M: Int32, N: Int32, TILE_SIZE: Int32) raises:
    pass
