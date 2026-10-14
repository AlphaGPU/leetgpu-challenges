from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K_cache, V_cache, K_tree, V_tree, parents, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K_cache: UnsafePointer[Float32, MutAnyOrigin],
    V_cache: UnsafePointer[Float32, MutAnyOrigin],
    K_tree: UnsafePointer[Float32, MutAnyOrigin],
    V_tree: UnsafePointer[Float32, MutAnyOrigin],
    parents: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    P: Int32,
    T: Int32,
    H: Int32,
    D: Int32,
) raises:
    pass
