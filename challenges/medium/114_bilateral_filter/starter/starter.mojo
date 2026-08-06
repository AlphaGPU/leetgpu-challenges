from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# image, output are device pointers
@export
def solve(
    image: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    H: Int32,
    W: Int32,
    spatial_sigma: Float32,
    range_sigma: Float32,
    radius: Int32,
) raises:
    pass
