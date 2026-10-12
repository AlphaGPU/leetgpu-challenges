from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K_int8: UnsafePointer[Int8, MutAnyOrigin],
    V_int8: UnsafePointer[Int8, MutAnyOrigin],
    k_scale: UnsafePointer[Float32, MutAnyOrigin],
    v_scale: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    num_heads: Int32,
    seq_len: Int32,
    head_dim: Int32,
) raises:
    pass
