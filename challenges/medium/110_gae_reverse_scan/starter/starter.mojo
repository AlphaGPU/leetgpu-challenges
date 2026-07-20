from std.memory import UnsafePointer


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
