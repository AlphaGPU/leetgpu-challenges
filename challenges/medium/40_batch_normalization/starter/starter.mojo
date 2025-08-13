from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# input, gamma, beta, output are device pointers
@export
def solve(input: UnsafePointer[Float32], gamma: UnsafePointer[Float32], 
          beta: UnsafePointer[Float32], output: UnsafePointer[Float32], 
          N: Int32, C: Int32, eps: Float32):
    pass
