from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Q, Cos, Sin, Output are device pointers
@export
def solve(Q: UnsafePointer[Float32], Cos: UnsafePointer[Float32], Sin: UnsafePointer[Float32], Output: UnsafePointer[Float32], M: Int32, D: Int32):
    pass