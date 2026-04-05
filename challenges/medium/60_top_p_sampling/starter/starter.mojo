from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.gpu import block_dim, block_idx, thread_idx

@export
fn solve(logits: UnsafePointer[Float32, MutExternalOrigin], p: UnsafePointer[Float32, MutExternalOrigin],
         seed: UnsafePointer[Int32, MutExternalOrigin], sampled_token: UnsafePointer[Int32, MutExternalOrigin],
         vocab_size: Int32) raises:
    pass
