from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv


# prompts, output, weights are device pointers
@export
def solve(
    prompts: UnsafePointer[Int32],
    output: UnsafePointer[Float32],
    weights: UnsafePointer[Float32],
    batch_size: Int32,
):
    pass
