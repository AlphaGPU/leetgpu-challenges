from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn count_3d_equal_kernel(input: UnsafePointer[Int32], output: UnsafePointer[Int32], N: Int32, M: Int32, K: Int32, P: Int32):
    pass

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Int32], output: UnsafePointer[Int32], N: Int32, M: Int32, K: Int32, P: Int32):
    var BLOCK_SIZE: Int32 = 8
    var ctx = DeviceContext()
    var grid_dim_x = ceildiv(K, BLOCK_SIZE)
    var grid_dim_y = ceildiv(M, BLOCK_SIZE)
    var grid_dim_z = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[count_3d_equal_kernel](
        input, output, N, M, K, P,
        grid_dim = (grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    )
    ctx.synchronize() 
