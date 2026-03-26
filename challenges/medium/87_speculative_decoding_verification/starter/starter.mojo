from gpu.host import DeviceContext
from memory import UnsafePointer

# draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
@export
def solve(
    draft_tokens: UnsafePointer[Int32],
    draft_probs: UnsafePointer[Float32],
    target_probs: UnsafePointer[Float32],
    uniform_samples: UnsafePointer[Float32],
    output_tokens: UnsafePointer[Int32],
    B: Int32,
    T: Int32,
    V: Int32,
):
    pass
