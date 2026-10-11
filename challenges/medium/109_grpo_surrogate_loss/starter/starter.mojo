from std.memory import UnsafePointer


# rewards, log_pi, log_pi_old, log_ref, output are device pointers
@export
def solve(
    rewards: UnsafePointer[Float32, MutAnyOrigin],
    log_pi: UnsafePointer[Float32, MutAnyOrigin],
    log_pi_old: UnsafePointer[Float32, MutAnyOrigin],
    log_ref: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    clip_eps: Float32,
    beta: Float32,
    B: Int32,
    G: Int32,
    S: Int32,
) raises:
    pass
