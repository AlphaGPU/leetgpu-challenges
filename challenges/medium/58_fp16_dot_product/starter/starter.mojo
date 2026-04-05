from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# A, B, result are device pointers
@export
fn solve(A: UnsafePointer[Float16, MutExternalOrigin], B: UnsafePointer[Float16, MutExternalOrigin], result: UnsafePointer[Float16, MutExternalOrigin], N: Int32) raises:
    pass
