from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    data_x: UnsafePointer[Float32, MutAnyOrigin],
    data_y: UnsafePointer[Float32, MutAnyOrigin],
    labels: UnsafePointer[Int32, MutAnyOrigin],
    initial_centroid_x: UnsafePointer[Float32, MutAnyOrigin],
    initial_centroid_y: UnsafePointer[Float32, MutAnyOrigin],
    final_centroid_x: UnsafePointer[Float32, MutAnyOrigin],
    final_centroid_y: UnsafePointer[Float32, MutAnyOrigin],
    sample_size: Int32,
    k: Int32,
    max_iterations: Int32,
) raises:
    pass
