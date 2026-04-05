from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# Q, cos, sin, output are device pointers
@export
fn solve(Q: UnsafePointer[Float32, MutExternalOrigin], cos: UnsafePointer[Float32, MutExternalOrigin], sin: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], M: Int32, D: Int32) raises:
    pass
