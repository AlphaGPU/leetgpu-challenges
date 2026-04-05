from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn matrix_add_kernel(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], C: UnsafePointer[Float32, MutExternalOrigin], N: Int32):
    pass

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], C: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var n_elements = N * N
    var num_blocks = ceildiv(n_elements, BLOCK_SIZE)

    var _kernel = ctx.compile_function[matrix_add_kernel, matrix_add_kernel]()
    ctx.enqueue_function(_kernel,
        A, B, C, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
