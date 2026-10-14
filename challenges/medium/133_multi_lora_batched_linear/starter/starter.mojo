from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, W, A, B, adapter_ids, y are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    W: UnsafePointer[Float32, MutAnyOrigin],
    A: UnsafePointer[Float32, MutAnyOrigin],
    B: UnsafePointer[Float32, MutAnyOrigin],
    adapter_ids: UnsafePointer[Int32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    T: Int32,
    K: Int32,
    N: Int32,
    R: Int32,
    L: Int32,
    scale: Float32,
) raises:
    pass
