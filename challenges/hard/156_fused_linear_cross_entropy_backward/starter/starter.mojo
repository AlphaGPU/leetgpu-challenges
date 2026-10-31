from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# X, W, targets, dX, dW are device pointers
@export
def solve(
    X: UnsafePointer[Float32, MutAnyOrigin],
    W: UnsafePointer[Float32, MutAnyOrigin],
    targets: UnsafePointer[Int32, MutAnyOrigin],
    dX: UnsafePointer[Float32, MutAnyOrigin],
    dW: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    D: Int32,
    V: Int32,
) raises:
    pass
