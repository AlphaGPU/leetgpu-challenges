from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# q, k_cache, v_cache, idx_q, idx_k, idx_w, output are device pointers
@export
def solve(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k_cache: UnsafePointer[Float32, MutAnyOrigin],
    v_cache: UnsafePointer[Float32, MutAnyOrigin],
    idx_q: UnsafePointer[Int8, MutAnyOrigin],
    idx_k: UnsafePointer[Int8, MutAnyOrigin],
    idx_w: UnsafePointer[Int8, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    S: Int32,
    H: Int32,
    D: Int32,
    H_I: Int32,
    D_I: Int32,
    topk: Int32,
) raises:
    pass
