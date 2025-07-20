from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn matrix_transpose_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], rows: Int32, cols: Int32):
    pass

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], rows: Int32, cols: Int32):
    var BLOCK_SIZE: Int32 = 32
    var ctx = DeviceContext()
    
    var grid_dim_x = ceildiv(cols, BLOCK_SIZE)
    var grid_dim_y = ceildiv(rows, BLOCK_SIZE)
    
    ctx.enqueue_function[matrix_transpose_kernel](
        input, output, rows, cols,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)
    )
    
    ctx.synchronize() 