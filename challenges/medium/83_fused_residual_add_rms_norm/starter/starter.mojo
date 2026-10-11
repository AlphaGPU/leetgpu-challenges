from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, residual, weight, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    residual: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
    eps: Float32,
) raises:
    pass
