from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn invert_kernel(image: UnsafePointer[UInt8, MutExternalOrigin], width: Int32, height: Int32):
    pass

# image is a device pointer (i.e. pointer to memory on the GPU)
@export
fn solve(image: UnsafePointer[UInt8, MutExternalOrigin], width: Int32, height: Int32) raises:
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()

    var total_pixels = width * height
    var blocksPerGrid = ceildiv(total_pixels, threadsPerBlock)

    var _kernel = ctx.compile_function[invert_kernel, invert_kernel]()
    ctx.enqueue_function(_kernel,
        image, width, height,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )

    ctx.synchronize()
