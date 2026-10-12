from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K_idx: UnsafePointer[UInt8, MutAnyOrigin],
    qjl_signs: UnsafePointer[Int8, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    Pi: UnsafePointer[Float32, MutAnyOrigin],
    S_mat: UnsafePointer[Float32, MutAnyOrigin],
    codebook: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    S: Int32,
    D: Int32,
    C: Int32,
) raises:
    pass
