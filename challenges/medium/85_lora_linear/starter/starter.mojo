from std.gpu.host import DeviceContext
from std.memory import UnsafePointer

# x, W, A, B, output are device pointers
@export
fn solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    W: UnsafePointer[Float32, MutExternalOrigin],
    A: UnsafePointer[Float32, MutExternalOrigin],
    B: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    batch: Int32,
    d_in: Int32,
    d_out: Int32,
    rank: Int32,
    lora_scale: Float32,
) raises:
    pass
