from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv

# signal, spectrum are device pointers
@export
fn solve(signal: UnsafePointer[Float32, MutExternalOrigin], spectrum: UnsafePointer[Float32, MutExternalOrigin], M: Int32, N: Int32) raises:
    pass
