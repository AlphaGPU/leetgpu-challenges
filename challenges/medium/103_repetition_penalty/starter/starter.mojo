from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# logits, input_ids are device pointers
@export
def solve(
    logits: UnsafePointer[Float32, MutExternalOrigin],
    input_ids: UnsafePointer[Int32, MutExternalOrigin],
    penalty: Float32,
    B: Int32,
    V: Int32,
    T: Int32,
) raises:
    pass
