from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# params, grad, m, v are device pointers
@export
def solve(
    params: UnsafePointer[Float32, MutExternalOrigin],
    grad: UnsafePointer[Float32, MutExternalOrigin],
    m: UnsafePointer[Float32, MutExternalOrigin],
    v: UnsafePointer[Float32, MutExternalOrigin],
    N: Int32,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    t: Int32,
) raises:
    pass
