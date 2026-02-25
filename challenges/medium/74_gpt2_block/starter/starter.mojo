from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# x, output, weights are device pointers
@export
def solve(x: UnsafePointer[Float32], output: UnsafePointer[Float32], weights: UnsafePointer[Float32], seq_len: Int32):
    pass
