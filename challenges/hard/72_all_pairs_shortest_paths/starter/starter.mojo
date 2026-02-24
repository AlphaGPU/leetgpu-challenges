from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# dist, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(dist: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    pass
