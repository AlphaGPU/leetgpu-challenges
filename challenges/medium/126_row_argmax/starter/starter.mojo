from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# logits, output are device pointers
@export
def solve(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Int32, MutAnyOrigin],
    B: Int32,
    V: Int32,
) raises:
    pass
