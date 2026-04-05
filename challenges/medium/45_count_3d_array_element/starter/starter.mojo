from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(input: UnsafePointer[Int32, MutExternalOrigin], output: UnsafePointer[Int32, MutExternalOrigin], N: Int32, M: Int32, K: Int32, P: Int32) raises:
    pass
