from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Q, K, V, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(Q: UnsafePointer[Float32], K: UnsafePointer[Float32], V: UnsafePointer[Float32], 
          output: UnsafePointer[Float32], M: Int32, d: Int32, window_size: Int32):
    pass 