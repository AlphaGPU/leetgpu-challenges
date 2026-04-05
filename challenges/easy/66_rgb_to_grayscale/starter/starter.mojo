from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn rgb_to_grayscale_kernel(input: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], width: Int32, height: Int32):
    pass

# input, output are device pointers
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], width: Int32, height: Int32) raises:
    var total_pixels = width * height
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(total_pixels, BLOCK_SIZE)

    var _kernel = ctx.compile_function[rgb_to_grayscale_kernel, rgb_to_grayscale_kernel]()
    ctx.enqueue_function(_kernel,
        input, output, width, height,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
