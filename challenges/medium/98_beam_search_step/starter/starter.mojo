from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens are device pointers
@export
def solve(
    beam_scores: UnsafePointer[Float32, MutAnyOrigin],
    token_logprobs: UnsafePointer[Float32, MutAnyOrigin],
    new_beam_scores: UnsafePointer[Float32, MutAnyOrigin],
    parent_beam_indices: UnsafePointer[Int32, MutAnyOrigin],
    next_tokens: UnsafePointer[Int32, MutAnyOrigin],
    B: Int32,
    K: Int32,
    V: Int32,
) raises:
    pass
