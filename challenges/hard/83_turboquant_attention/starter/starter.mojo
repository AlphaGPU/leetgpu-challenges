from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores are device pointers
@export
def solve(
    Q: UnsafePointer[Float32],
    K_idx: UnsafePointer[UInt8],
    qjl_signs: UnsafePointer[Int8],
    gamma: UnsafePointer[Float32],
    Pi: UnsafePointer[Float32],
    S_mat: UnsafePointer[Float32],
    codebook: UnsafePointer[Float32],
    scores: UnsafePointer[Float32],
    B: Int32,
    S: Int32,
    D: Int32,
    C: Int32,
):
    pass