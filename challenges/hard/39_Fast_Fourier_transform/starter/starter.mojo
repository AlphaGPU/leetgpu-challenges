from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# signal and spectrum are device pointers
@export
def solve(signal: UnsafePointer[Float32], spectrum: UnsafePointer[Float32], N: Int32):
    pass