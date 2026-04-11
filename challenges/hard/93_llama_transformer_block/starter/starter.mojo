from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, output, weights, cos, sin are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    weights: UnsafePointer[Float32, MutExternalOrigin],
    cos: UnsafePointer[Float32, MutExternalOrigin],
    sin: UnsafePointer[Float32, MutExternalOrigin],
    seq_len: Int32,
) raises:
    pass
