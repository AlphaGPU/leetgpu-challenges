from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv


# x, W_gate, W_up, W_down, output are device pointers
@export
def solve(
    x: UnsafePointer[Float32],
    W_gate: UnsafePointer[Float32],
    W_up: UnsafePointer[Float32],
    W_down: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    M: Int32,
    d_model: Int32,
    d_ffn: Int32,
):
    pass
