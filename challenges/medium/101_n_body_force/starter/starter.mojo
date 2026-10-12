from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# positions, masses, forces are device pointers
@export
def solve(
    positions: UnsafePointer[Float32, MutAnyOrigin],
    masses: UnsafePointer[Float32, MutAnyOrigin],
    forces: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
) raises:
    pass
