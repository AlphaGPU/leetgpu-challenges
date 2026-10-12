from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


@export
def solve(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    true_labels: UnsafePointer[Int32, MutAnyOrigin],
    loss: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    C: Int32,
) raises:
    pass
