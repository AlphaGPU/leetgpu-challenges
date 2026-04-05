from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(predictions: UnsafePointer[Float32, MutExternalOrigin], targets: UnsafePointer[Float32, MutExternalOrigin], mse: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    pass
