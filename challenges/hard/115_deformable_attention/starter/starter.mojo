from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# value, spatial_shapes, sampling_loc, attn_weight, output are device pointers
@export
def solve(
    value: UnsafePointer[Float32, MutExternalOrigin],
    spatial_shapes: UnsafePointer[Int32, MutExternalOrigin],
    sampling_loc: UnsafePointer[Float32, MutExternalOrigin],
    attn_weight: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    N: Int32,
    S: Int32,
    Q: Int32,
    H: Int32,
    D: Int32,
    L: Int32,
    P: Int32,
) raises:
    pass
