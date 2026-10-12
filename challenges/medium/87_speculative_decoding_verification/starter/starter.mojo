from max.gpu.host import DeviceContext
from std.memory import UnsafePointer


# draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
@export
def solve(
    draft_tokens: UnsafePointer[Int32, MutAnyOrigin],
    draft_probs: UnsafePointer[Float32, MutAnyOrigin],
    target_probs: UnsafePointer[Float32, MutAnyOrigin],
    uniform_samples: UnsafePointer[Float32, MutAnyOrigin],
    output_tokens: UnsafePointer[Int32, MutAnyOrigin],
    B: Int32,
    T: Int32,
    V: Int32,
):
    pass
