from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# images, patch_weight, patch_bias, cls_token, pos_embed, output are device pointers
@export
def solve(
    images: UnsafePointer[Float32, MutExternalOrigin],
    patch_weight: UnsafePointer[Float32, MutExternalOrigin],
    patch_bias: UnsafePointer[Float32, MutExternalOrigin],
    cls_token: UnsafePointer[Float32, MutExternalOrigin],
    pos_embed: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    B: Int32,
    C: Int32,
    H: Int32,
    W: Int32,
    P: Int32,
    D: Int32,
) raises:
    pass
