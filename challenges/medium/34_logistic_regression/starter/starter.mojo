from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# X, y, beta are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(X: UnsafePointer[Float32, MutExternalOrigin], y: UnsafePointer[Float32, MutExternalOrigin], beta: UnsafePointer[Float32, MutExternalOrigin], n_samples: Int32, n_features: Int32) raises:
    pass
