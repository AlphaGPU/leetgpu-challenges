from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn sigmoid_kernel(X: UnsafePointer[Float32], Y: UnsafePointer[Float32], N: Int32):
    pass

# X, Y are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(X: UnsafePointer[Float32], Y: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[sigmoid_kernel](
        X, Y, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
