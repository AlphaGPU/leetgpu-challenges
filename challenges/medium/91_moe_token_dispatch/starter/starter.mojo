from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, expert_idx, dispatched_x, token_counts are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    expert_idx: UnsafePointer[Int32, MutExternalOrigin],
    dispatched_x: UnsafePointer[Float32, MutExternalOrigin],
    token_counts: UnsafePointer[Int32, MutExternalOrigin],
    T: Int32,
    D: Int32,
    E: Int32,
    capacity: Int32,
) raises:
    pass
