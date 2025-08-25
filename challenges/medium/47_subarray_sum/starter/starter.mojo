from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn subarray_sum_kernel(input: UnsafePointer[Int32], output: UnsafePointer[Int32], N: Int32, S: Int32, E: Int32):
    pass

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Int32], output: UnsafePointer[Int32], N: Int32, S: Int32, E: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[subarray_sum_kernel](
        input, output, N, S, E,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
