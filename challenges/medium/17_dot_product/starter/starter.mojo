from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], result: UnsafePointer[Float32], N: Int32):
    pass 