from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# input, gamma, beta, output are device pointers
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], gamma: UnsafePointer[Float32, MutExternalOrigin],
          beta: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin],
          N: Int32, C: Int32, eps: Float32) raises:
    pass
