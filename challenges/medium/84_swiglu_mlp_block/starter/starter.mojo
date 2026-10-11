from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, W_gate, W_up, W_down, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    W_gate: UnsafePointer[Float32, MutAnyOrigin],
    W_up: UnsafePointer[Float32, MutAnyOrigin],
    W_down: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    M: Int32,
    d_model: Int32,
    d_ffn: Int32,
):
    pass
