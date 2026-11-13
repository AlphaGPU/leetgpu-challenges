from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# image_embeds, text_embeds, loss are device pointers
@export
def solve(
    image_embeds: UnsafePointer[Float32, MutAnyOrigin],
    text_embeds: UnsafePointer[Float32, MutAnyOrigin],
    loss: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    D: Int32,
    logit_scale: Float32,
) raises:
    pass
