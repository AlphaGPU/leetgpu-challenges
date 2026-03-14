from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# a, x, h are device pointers
@export
def solve(a: UnsafePointer[Float32], x: UnsafePointer[Float32], h: UnsafePointer[Float32], B: Int32, L: Int32):
    pass
