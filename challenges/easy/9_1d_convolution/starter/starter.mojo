from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn convolution_1d_kernel(input: UnsafePointer[Float32, MutExternalOrigin], kernel: UnsafePointer[Float32, MutExternalOrigin],
                         output: UnsafePointer[Float32, MutExternalOrigin], input_size: Int32, kernel_size: Int32):
    pass

# input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], kernel: UnsafePointer[Float32, MutExternalOrigin],
          output: UnsafePointer[Float32, MutExternalOrigin], input_size: Int32, kernel_size: Int32) raises:
    var output_size = input_size - kernel_size + 1
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()

    var blocksPerGrid = ceildiv(output_size, threadsPerBlock)

    var _kernel = ctx.compile_function[convolution_1d_kernel, convolution_1d_kernel]()
    ctx.enqueue_function(_kernel,
        input, kernel, output, input_size, kernel_size,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )

    ctx.synchronize()
