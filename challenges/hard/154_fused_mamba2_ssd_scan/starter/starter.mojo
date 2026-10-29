from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, dt, dt_bias, A, B, C, D, z, y are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dt: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    A: UnsafePointer[Float32, MutAnyOrigin],
    B: UnsafePointer[Float32, MutAnyOrigin],
    C: UnsafePointer[Float32, MutAnyOrigin],
    D: UnsafePointer[Float32, MutAnyOrigin],
    z: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    batch: Int32,
    seq_len: Int32,
    num_heads: Int32,
    head_dim: Int32,
    num_groups: Int32,
    state_dim: Int32,
) raises:
    pass
