from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], C: UnsafePointer[Float32, MutExternalOrigin], BATCH: Int32, M: Int32, N: Int32, K: Int32) raises:
    pass
