from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    agents: UnsafePointer[Float32, MutAnyOrigin],
    agents_next: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
) raises:
    pass
