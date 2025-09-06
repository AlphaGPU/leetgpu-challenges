from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn prime_checker_kernel(n: Int32, output: UnsafePointer[Int32]):
    pass

# output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(n: Int32, output: UnsafePointer[Int32]):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    ctx.enqueue_function[prime_checker_kernel](
        n, output,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()