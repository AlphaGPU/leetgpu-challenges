from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from max.gpu import block_dim, block_idx, thread_idx


@export
def solve(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    p: UnsafePointer[Float32, MutAnyOrigin],
    seed: UnsafePointer[Int32, MutAnyOrigin],
    sampled_token: UnsafePointer[Int32, MutAnyOrigin],
    vocab_size: Int32,
) raises:
    pass
