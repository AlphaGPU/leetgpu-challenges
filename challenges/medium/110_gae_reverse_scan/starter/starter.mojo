from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# rewards, values, advantages are device pointers
@export
def solve(
    rewards: UnsafePointer[Float32, MutAnyOrigin],
    values: UnsafePointer[Float32, MutAnyOrigin],
    advantages: UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32,
    lam: Float32,
    B: Int32,
    S: Int32,
) raises:
    pass
