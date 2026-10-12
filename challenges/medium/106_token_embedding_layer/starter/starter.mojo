from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device pointers
@export
def solve(
    token_ids: UnsafePointer[Int32, MutAnyOrigin],
    position_ids: UnsafePointer[Int32, MutAnyOrigin],
    token_embeddings: UnsafePointer[Float32, MutAnyOrigin],
    position_embeddings: UnsafePointer[Float32, MutAnyOrigin],
    gamma: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    T: Int32,
    V: Int32,
    P: Int32,
    D: Int32,
    eps: Float32,
) raises:
    pass
