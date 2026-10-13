from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, topk_ids, topk_weights, w_gate, w_up, w_down, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    topk_ids: UnsafePointer[Int32, MutAnyOrigin],
    topk_weights: UnsafePointer[Float32, MutAnyOrigin],
    w_gate: UnsafePointer[Float32, MutAnyOrigin],
    w_up: UnsafePointer[Float32, MutAnyOrigin],
    w_down: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    T: Int32,
    H: Int32,
    I: Int32,
    E: Int32,
    K: Int32,
) raises:
    pass
