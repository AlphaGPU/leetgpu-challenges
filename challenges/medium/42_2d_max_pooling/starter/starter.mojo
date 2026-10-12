from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# input, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
    H: Int32,
    W: Int32,
    kernel_size: Int32,
    stride: Int32,
    padding: Int32,
) raises:
    pass
