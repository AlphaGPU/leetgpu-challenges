from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn sigmoid_kernel(X: UnsafePointer[Float32, MutExternalOrigin], Y: UnsafePointer[Float32, MutExternalOrigin], N: Int32):
    pass

# X, Y are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(X: UnsafePointer[Float32, MutExternalOrigin], Y: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    var _kernel = ctx.compile_function[sigmoid_kernel, sigmoid_kernel]()
    ctx.enqueue_function(_kernel,
        X, Y, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
