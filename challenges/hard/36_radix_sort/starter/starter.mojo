from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer

# input, output are device pointers
@export
fn solve(input: UnsafePointer[UInt32, MutExternalOrigin], output: UnsafePointer[UInt32, MutExternalOrigin], N: Int32) raises:
    pass
