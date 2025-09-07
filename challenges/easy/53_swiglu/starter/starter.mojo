from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv


fn swiglu_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    pass


# input, output are device pointers
@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N // 2, BLOCK_SIZE)
    
    ctx.enqueue_function[swiglu_kernel](
        input, output, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )
    
    ctx.synchronize()
