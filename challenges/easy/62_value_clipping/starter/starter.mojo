from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn clip_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32, lo: Float32, hi: Float32):
    pass


# input, output are device pointers
@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32, lo: Float32, hi: Float32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[clip_kernel](
        input, output, N, lo, hi,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
