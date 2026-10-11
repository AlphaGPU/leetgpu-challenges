from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# prompts, output, weights are device pointers
@export
def solve(
    prompts: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    weights: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int32,
):
    pass
