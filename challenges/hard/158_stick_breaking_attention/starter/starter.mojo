from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# q, k, v, output are device pointers
@export
def solve(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    H: Int32,
    H_KV: Int32,
    S: Int32,
    D: Int32,
) raises:
    pass
