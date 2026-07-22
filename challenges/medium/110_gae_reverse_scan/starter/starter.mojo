from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# rewards, values, advantages are device pointers
@export
def solve(
    rewards: UnsafePointer[Float32, MutExternalOrigin],
    values: UnsafePointer[Float32, MutExternalOrigin],
    advantages: UnsafePointer[Float32, MutExternalOrigin],
    gamma: Float32,
    lam: Float32,
    B: Int32,
    S: Int32,
) raises:
    pass
