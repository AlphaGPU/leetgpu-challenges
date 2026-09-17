from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# boxes, scores, keep are device pointers
@export
def solve(
    boxes: UnsafePointer[Float32, MutExternalOrigin],
    scores: UnsafePointer[Float32, MutExternalOrigin],
    keep: UnsafePointer[Int32, MutExternalOrigin],
    N: Int32,
    iou_threshold: Float32,
) raises:
    pass
