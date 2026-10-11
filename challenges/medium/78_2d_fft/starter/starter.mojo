from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# signal, spectrum are device pointers
@export
def solve(
    signal: UnsafePointer[Float32, MutAnyOrigin],
    spectrum: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    N: Int32,
) raises:
    pass
