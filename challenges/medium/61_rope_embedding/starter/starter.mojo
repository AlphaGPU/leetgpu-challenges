from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Q, cos, sin, output are device pointers
@export
def solve(Q: UnsafePointer[Float32], cos: UnsafePointer[Float32], sin: UnsafePointer[Float32], output: UnsafePointer[Float32], M: Int32, D: Int32):
    pass