from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(A: UnsafePointer[Float32], x: UnsafePointer[Float32], y: UnsafePointer[Float32], M: Int32, N: Int32, nnz: Int32):
    pass 