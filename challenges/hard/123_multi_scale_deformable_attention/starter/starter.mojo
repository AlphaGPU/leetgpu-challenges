from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# value, spatial_shapes, sampling_loc, attn_weight, output are device pointers
@export
def solve(
    value: UnsafePointer[Float32, MutAnyOrigin],
    spatial_shapes: UnsafePointer[Int32, MutAnyOrigin],
    sampling_loc: UnsafePointer[Float32, MutAnyOrigin],
    attn_weight: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    num_queries: Int32,
    num_heads: Int32,
    head_dim: Int32,
    num_levels: Int32,
    num_points: Int32,
) raises:
    pass
