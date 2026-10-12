from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K_prefix, V_prefix, K_suffix, V_suffix, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K_prefix: UnsafePointer[Float32, MutAnyOrigin],
    V_prefix: UnsafePointer[Float32, MutAnyOrigin],
    K_suffix: UnsafePointer[Float32, MutAnyOrigin],
    V_suffix: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int32,
    num_heads: Int32,
    prefix_len: Int32,
    suffix_len: Int32,
    head_dim: Int32,
) raises:
    pass
