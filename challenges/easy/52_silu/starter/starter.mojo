from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


def silu_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
):
    pass


# input, output are device pointers
@export
def solve(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
) raises:
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()

    var blocksPerGrid = ceildiv(N, threadsPerBlock)

    var _kernel = ctx.compile_function[silu_kernel]()
    ctx.enqueue_function(
        _kernel, input, output, N, grid_dim=blocksPerGrid, block_dim=threadsPerBlock
    )

    ctx.synchronize()
