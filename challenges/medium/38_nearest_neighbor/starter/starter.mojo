from std.memory import UnsafePointer


# points and indices are device pointers
@export
def solve(
    points: UnsafePointer[Float32, MutAnyOrigin],
    indices: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
) raises:
    pass
