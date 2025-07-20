from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

# input, output are device pointers
@export                         
def solve(input: UnsafePointer[UInt32], output: UnsafePointer[UInt32], N: Int32):
    pass 