from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# A, B, result are device pointers
@export
def solve(A: UnsafePointer[Float16], B: UnsafePointer[Float16], result: UnsafePointer[Float16], N: Int32):
    pass
