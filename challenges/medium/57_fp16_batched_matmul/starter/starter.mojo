from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(A: UnsafePointer[Float16], B: UnsafePointer[Float16], C: UnsafePointer[Float16], BATCH: Int32, M: Int32, N: Int32, K: Int32):
    pass
