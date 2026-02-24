from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv


# image, output are device pointers
@export
def solve(
    image: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    H: Int32,
    W: Int32,
    spatial_sigma: Float32,
    range_sigma: Float32,
    radius: Int32,
):
    pass
