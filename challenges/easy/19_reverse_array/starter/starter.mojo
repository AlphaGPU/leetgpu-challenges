from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn reverse_array_kernel(input: UnsafePointer[Float32, MutExternalOrigin], N: Int32):
    pass

# input is a device pointer (i.e. pointer to memory on the GPU)
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()

    var blocksPerGrid = ceildiv(N, threadsPerBlock)

    var _kernel = ctx.compile_function[reverse_array_kernel, reverse_array_kernel]()
    ctx.enqueue_function(_kernel,
        input, N,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )

    ctx.synchronize()
