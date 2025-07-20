from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(input: UnsafePointer[Int32], histogram: UnsafePointer[Int32], N: Int32, num_bins: Int32):
    pass 