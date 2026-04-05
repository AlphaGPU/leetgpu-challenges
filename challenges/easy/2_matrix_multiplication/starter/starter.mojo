from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn matrix_multiplication_kernel(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], C: UnsafePointer[Float32, MutExternalOrigin], M: Int32, N: Int32, K: Int32):
    pass

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], C: UnsafePointer[Float32, MutExternalOrigin], M: Int32, N: Int32, K: Int32) raises:
    var BLOCK_SIZE: Int32 = 16
    var ctx = DeviceContext()

    var grid_dim_x = ceildiv(K, BLOCK_SIZE)
    var grid_dim_y = ceildiv(M, BLOCK_SIZE)

    var _kernel = ctx.compile_function[matrix_multiplication_kernel, matrix_multiplication_kernel]()
    ctx.enqueue_function(_kernel,
        A, B, C, M, N, K,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)
    )

    ctx.synchronize()
