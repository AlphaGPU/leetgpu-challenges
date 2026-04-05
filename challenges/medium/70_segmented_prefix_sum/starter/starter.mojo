from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# values, flags, output are device pointers
@export
fn solve(values: UnsafePointer[Float32, MutExternalOrigin], flags: UnsafePointer[Int32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    pass
