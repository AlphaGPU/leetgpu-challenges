from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer


# input, output are device pointers
@export
def solve(
    input: UnsafePointer[UInt32, MutAnyOrigin],
    output: UnsafePointer[UInt32, MutAnyOrigin],
    N: Int32,
) raises:
    pass
