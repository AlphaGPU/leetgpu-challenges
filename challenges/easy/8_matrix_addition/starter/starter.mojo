from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn matrix_add_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    pass

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var n_elements = N * N
    var num_blocks = ceildiv(n_elements, BLOCK_SIZE)

    ctx.enqueue_function[matrix_add_kernel](
        A, B, C, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
