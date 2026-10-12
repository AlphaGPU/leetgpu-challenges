from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# boxes, scores, keep are device pointers
@export
def solve(
    boxes: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    keep: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    iou_threshold: Float32,
) raises:
    pass
