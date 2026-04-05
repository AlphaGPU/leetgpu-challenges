from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# input, output are device pointers
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], gamma: Float32,
          beta: Float32, output: UnsafePointer[Float32, MutExternalOrigin],
          N: Int32, eps: Float32) raises:
    pass
