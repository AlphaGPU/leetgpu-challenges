from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, W_gate_up, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    W_gate_up: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    H: Int32,
    I: Int32,
) raises:
    pass
