from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# a, x, h are device pointers
@export
fn solve(a: UnsafePointer[Float32, MutExternalOrigin], x: UnsafePointer[Float32, MutExternalOrigin], h: UnsafePointer[Float32, MutExternalOrigin], B: Int32, L: Int32) raises:
    pass
