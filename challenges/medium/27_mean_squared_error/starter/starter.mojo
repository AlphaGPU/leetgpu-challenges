from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    predictions: UnsafePointer[Float32, MutAnyOrigin],
    targets: UnsafePointer[Float32, MutAnyOrigin],
    mse: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
) raises:
    pass
