from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# input, output are device pointers
@export
def solve(
    input: UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32,
    beta: Float32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    eps: Float32,
) raises:
    pass
