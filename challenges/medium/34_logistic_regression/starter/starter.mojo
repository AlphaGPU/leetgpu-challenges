from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# X, y, beta are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    X: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    n_samples: Int32,
    n_features: Int32,
) raises:
    pass
