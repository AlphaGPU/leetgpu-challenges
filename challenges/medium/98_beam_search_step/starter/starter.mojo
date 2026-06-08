from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens are device pointers
@export
def solve(
    beam_scores: UnsafePointer[Float32, MutExternalOrigin],
    token_logprobs: UnsafePointer[Float32, MutExternalOrigin],
    new_beam_scores: UnsafePointer[Float32, MutExternalOrigin],
    parent_beam_indices: UnsafePointer[Int32, MutExternalOrigin],
    next_tokens: UnsafePointer[Int32, MutExternalOrigin],
    B: Int32,
    K: Int32,
    V: Int32,
) raises:
    pass
