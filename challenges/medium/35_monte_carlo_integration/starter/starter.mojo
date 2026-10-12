from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# y_samples, result are device pointers
@export
def solve(
    y_samples: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
    a: Float32,
    b: Float32,
    n_samples: Int32,
) raises:
    pass
