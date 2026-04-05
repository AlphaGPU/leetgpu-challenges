from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], x: UnsafePointer[Float32, MutExternalOrigin], y: UnsafePointer[Float32, MutExternalOrigin], M: Int32, N: Int32, nnz: Int32) raises:
    pass
