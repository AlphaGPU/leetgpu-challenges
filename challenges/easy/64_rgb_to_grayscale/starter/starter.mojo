from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn rgb_to_grayscale_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], width: Int32, height: Int32):
    pass

# input, output are device pointers
@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], width: Int32, height: Int32):
    var total_pixels = width * height
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(total_pixels, BLOCK_SIZE)

    ctx.enqueue_function[rgb_to_grayscale_kernel](
        input, output, width, height,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()