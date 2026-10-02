from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty, output are device pointers
@export
def solve(
    logits: UnsafePointer[Float32, MutExternalOrigin],
    prompt_tokens: UnsafePointer[Int32, MutExternalOrigin],
    output_tokens: UnsafePointer[Int32, MutExternalOrigin],
    presence_penalty: UnsafePointer[Float32, MutExternalOrigin],
    frequency_penalty: UnsafePointer[Float32, MutExternalOrigin],
    repetition_penalty: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    B: Int32,
    V: Int32,
    P: Int32,
    G: Int32,
) raises:
    pass
