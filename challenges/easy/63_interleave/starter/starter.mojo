from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

fn interleave_kernel(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], N: Int32):
    pass

# A, B, output are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], output: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    var _kernel = ctx.compile_function[interleave_kernel, interleave_kernel]()
    ctx.enqueue_function(_kernel,
        A, B, output, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
