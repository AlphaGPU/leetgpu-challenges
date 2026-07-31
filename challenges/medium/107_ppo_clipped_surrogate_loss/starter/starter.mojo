from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# advantages, log_pi, log_pi_old, output are device pointers
@export
def solve(
    advantages: UnsafePointer[Float32, MutExternalOrigin],
    log_pi: UnsafePointer[Float32, MutExternalOrigin],
    log_pi_old: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    clip_eps: Float32,
    B: Int32,
    S: Int32,
) raises:
    pass
