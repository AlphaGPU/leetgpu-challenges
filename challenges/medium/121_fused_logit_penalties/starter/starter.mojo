from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty, output are device pointers
@export
def solve(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    prompt_tokens: UnsafePointer[Int32, MutAnyOrigin],
    output_tokens: UnsafePointer[Int32, MutAnyOrigin],
    presence_penalty: UnsafePointer[Float32, MutAnyOrigin],
    frequency_penalty: UnsafePointer[Float32, MutAnyOrigin],
    repetition_penalty: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    V: Int32,
    P: Int32,
    G: Int32,
) raises:
    pass
