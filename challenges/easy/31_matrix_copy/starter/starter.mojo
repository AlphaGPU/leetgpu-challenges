from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn copy_matrix_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], N: Int32):
    pass

# A, B are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], N: Int32):
    var total = N * N
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var blocksPerGrid = ceildiv(total, threadsPerBlock)
    
    ctx.enqueue_function[copy_matrix_kernel](
        A, B, N,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 