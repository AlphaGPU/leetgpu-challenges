from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# Q, K, V, output are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(Q: UnsafePointer[Float32, MutExternalOrigin], K: UnsafePointer[Float32, MutExternalOrigin], V: UnsafePointer[Float32, MutExternalOrigin],
          output: UnsafePointer[Float32, MutExternalOrigin], M: Int32, d: Int32) raises:
    pass
