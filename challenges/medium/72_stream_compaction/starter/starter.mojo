from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# A, output are device pointers
@export
def solve(
    A: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    output: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    pass
