from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(logits: UnsafePointer[Float32, MutExternalOrigin], topk_weights: UnsafePointer[Float32, MutExternalOrigin], topk_indices: UnsafePointer[Int32, MutExternalOrigin], M: Int32, E: Int32, k: Int32) raises:
    pass
