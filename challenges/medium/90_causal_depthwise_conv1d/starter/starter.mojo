from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# x, weight, bias, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    bias: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    L: Int32,
    D: Int32,
    K: Int32,
):
    pass
