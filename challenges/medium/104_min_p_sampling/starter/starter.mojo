from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# logits, probs are device pointers
@export
def solve(
    logits: UnsafePointer[Float32, MutExternalOrigin],
    probs: UnsafePointer[Float32, MutExternalOrigin],
    min_p: Float32,
    B: Int32,
    V: Int32,
) raises:
    pass
