from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(Q: UnsafePointer[Float32], K: UnsafePointer[Float32], V: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32, d_model: Int32, h: Int32):
    pass 