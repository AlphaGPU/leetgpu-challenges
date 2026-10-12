from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, output, weights are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    weights: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int32,
) raises:
    pass
