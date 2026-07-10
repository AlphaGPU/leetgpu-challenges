from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv


# x, W_qkv, Q, K, V are device pointers
@export
def solve(
    x: UnsafePointer[Float32],
    W_qkv: UnsafePointer[Float32],
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    M: Int32,
    num_heads: Int32,
    head_dim: Int32,
):
    pass
