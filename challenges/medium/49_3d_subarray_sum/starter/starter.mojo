from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Int32], output: UnsafePointer[Int32], N: Int32, M: Int32, K: Int32, S_DEP: Int32, E_DEP: Int32, S_ROW: Int32, E_ROW: Int32, S_COL: Int32, E_COL: Int32):
    pass
