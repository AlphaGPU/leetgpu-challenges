from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# partial_out, partial_lse, output are device pointers
@export
def solve(
    partial_out: UnsafePointer[Float32, MutExternalOrigin],
    partial_lse: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    num_splits: Int32,
    num_heads: Int32,
    head_dim: Int32,
) raises:
    pass
