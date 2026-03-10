from gpu.host import DeviceContext
from memory import UnsafePointer

# Q, K, V, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    num_q_heads: Int32,
    num_kv_heads: Int32,
    seq_len: Int32,
    head_dim: Int32,
):
    pass
