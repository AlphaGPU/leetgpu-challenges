from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn matrix_transpose_kernel(input: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], rows: Int32, cols: Int32):
    pass

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(input: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], rows: Int32, cols: Int32) raises:
    var BLOCK_SIZE: Int32 = 32
    var ctx = DeviceContext()

    var grid_dim_x = ceildiv(cols, BLOCK_SIZE)
    var grid_dim_y = ceildiv(rows, BLOCK_SIZE)

    var _kernel = ctx.compile_function[matrix_transpose_kernel, matrix_transpose_kernel]()
    ctx.enqueue_function(_kernel,
        input, output, rows, cols,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)
    )

    ctx.synchronize()
