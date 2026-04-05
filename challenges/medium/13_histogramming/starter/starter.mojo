from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(input: UnsafePointer[Int32, MutExternalOrigin], histogram: UnsafePointer[Int32, MutExternalOrigin], N: Int32, num_bins: Int32) raises:
    pass
