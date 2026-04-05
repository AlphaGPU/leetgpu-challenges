from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(logits: UnsafePointer[Float32, MutExternalOrigin], true_labels: UnsafePointer[Int32, MutExternalOrigin], loss: UnsafePointer[Float32, MutExternalOrigin], N: Int32, C: Int32) raises:
    pass
