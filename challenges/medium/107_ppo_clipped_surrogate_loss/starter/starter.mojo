from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# advantages, log_pi, log_pi_old, output are device pointers
@export
def solve(
    advantages: UnsafePointer[Float32, MutAnyOrigin],
    log_pi: UnsafePointer[Float32, MutAnyOrigin],
    log_pi_old: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    clip_eps: Float32,
    B: Int32,
    S: Int32,
) raises:
    pass
