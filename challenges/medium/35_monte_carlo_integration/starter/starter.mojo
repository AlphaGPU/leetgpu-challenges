from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# y_samples, result are device pointers
@export
fn solve(y_samples: UnsafePointer[Float32, MutExternalOrigin], result: UnsafePointer[Float32, MutExternalOrigin], a: Float32, b: Float32, n_samples: Int32) raises:
    pass
