from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin],
          N: Int32, C: Int32, H: Int32, W: Int32,
          kernel_size: Int32, stride: Int32, padding: Int32) raises:
    pass
