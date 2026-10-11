from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# input, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    input: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    S: Int32,
    E: Int32,
) raises:
    pass
