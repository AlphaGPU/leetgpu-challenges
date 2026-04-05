from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

@export
fn solve(agents: UnsafePointer[Float32, MutExternalOrigin], agents_next: UnsafePointer[Float32, MutExternalOrigin], N: Int32) raises:
    pass
