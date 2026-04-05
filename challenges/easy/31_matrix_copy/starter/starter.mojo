from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn copy_matrix_kernel(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], N: Int32):
    pass

# A, B are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    var total = N * N
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()

    var blocksPerGrid = ceildiv(total, threadsPerBlock)

    var _kernel = ctx.compile_function[copy_matrix_kernel, copy_matrix_kernel]()
    ctx.enqueue_function(_kernel,
        A, B, N,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )

    ctx.synchronize()
