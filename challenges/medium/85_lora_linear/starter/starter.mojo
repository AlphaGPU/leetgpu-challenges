from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# x, W, A, B, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    W: UnsafePointer[Float32, MutAnyOrigin],
    A: UnsafePointer[Float32, MutAnyOrigin],
    B: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch: Int32,
    d_in: Int32,
    d_out: Int32,
    rank: Int32,
    lora_scale: Float32,
) raises:
    pass
