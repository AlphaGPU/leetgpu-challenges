from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn invert_kernel(image: UnsafePointer[UInt8], width: Int32, height: Int32):
    pass

# image is a device pointer (i.e. pointer to memory on the GPU)
@export                         
def solve(image: UnsafePointer[UInt8], width: Int32, height: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var total_pixels = width * height
    var blocksPerGrid = ceildiv(total_pixels, threadsPerBlock)
    
    ctx.enqueue_function[invert_kernel](
        image, width, height,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 