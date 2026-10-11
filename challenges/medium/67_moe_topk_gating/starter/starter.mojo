from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    topk_weights: UnsafePointer[Float32, MutAnyOrigin],
    topk_indices: UnsafePointer[Int32, MutAnyOrigin],
    M: Int32,
    E: Int32,
    k: Int32,
) raises:
    pass
