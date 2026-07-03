from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device pointers
@export
def solve(
    token_ids: UnsafePointer[Int32, MutExternalOrigin],
    position_ids: UnsafePointer[Int32, MutExternalOrigin],
    token_embeddings: UnsafePointer[Float32, MutExternalOrigin],
    position_embeddings: UnsafePointer[Float32, MutExternalOrigin],
    gamma: UnsafePointer[Float32, MutExternalOrigin],
    beta: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    B: Int32,
    T: Int32,
    V: Int32,
    P: Int32,
    D: Int32,
    eps: Float32,
) raises:
    pass
