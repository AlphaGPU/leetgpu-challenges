from std.memory import UnsafePointer


# points and indices are device pointers
@export
def solve(
    points: UnsafePointer[Float32, MutExternalOrigin],
    indices: UnsafePointer[Int32, MutExternalOrigin],
    N: Int32,
) raises:
    pass
