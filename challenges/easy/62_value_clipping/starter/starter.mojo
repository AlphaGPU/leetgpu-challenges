from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


def clip_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    lo: Float32,
    hi: Float32,
    N: Int32,
):
    pass


# input, output are device pointers
@export
def solve(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    lo: Float32,
    hi: Float32,
    N: Int32,
) raises:
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    var _kernel = ctx.compile_function[clip_kernel]()
    ctx.enqueue_function(
        _kernel, input, output, lo, hi, N, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )

    ctx.synchronize()
