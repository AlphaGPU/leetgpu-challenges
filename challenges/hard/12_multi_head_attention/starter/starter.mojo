from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(Q: UnsafePointer[Float32, MutExternalOrigin], K: UnsafePointer[Float32, MutExternalOrigin], V: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], N: Int32, d_model: Int32, h: Int32) raises:
    pass
