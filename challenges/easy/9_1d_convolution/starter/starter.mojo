from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn convolution_1d_kernel(input: UnsafePointer[Float32], kernel: UnsafePointer[Float32], 
                         output: UnsafePointer[Float32], input_size: Int32, kernel_size: Int32):
    pass

# input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Float32], kernel: UnsafePointer[Float32], 
          output: UnsafePointer[Float32], input_size: Int32, kernel_size: Int32):
    var output_size = input_size - kernel_size + 1
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var blocksPerGrid = ceildiv(output_size, threadsPerBlock)
    
    ctx.enqueue_function[convolution_1d_kernel](
        input, kernel, output, input_size, kernel_size,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 