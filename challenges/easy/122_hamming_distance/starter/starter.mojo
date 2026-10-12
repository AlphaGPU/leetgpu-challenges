from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


def hamming_distance_kernel(
    codes: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Int32, MutAnyOrigin],
    query: Int32,
    N: Int32,
):
    pass


# codes, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    codes: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Int32, MutAnyOrigin],
    query: Int32,
    N: Int32,
) raises:
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    var _kernel = ctx.compile_function[hamming_distance_kernel]()
    ctx.enqueue_function(
        _kernel, codes, output, query, N, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )

    ctx.synchronize()
