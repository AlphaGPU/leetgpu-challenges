from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# images, patch_weight, patch_bias, cls_token, pos_embed, output are device pointers
@export
def solve(
    images: UnsafePointer[Float32, MutAnyOrigin],
    patch_weight: UnsafePointer[Float32, MutAnyOrigin],
    patch_bias: UnsafePointer[Float32, MutAnyOrigin],
    cls_token: UnsafePointer[Float32, MutAnyOrigin],
    pos_embed: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int32,
    C: Int32,
    H: Int32,
    W: Int32,
    P: Int32,
    D: Int32,
) raises:
    pass
