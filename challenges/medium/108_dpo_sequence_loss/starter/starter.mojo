from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
@export
def solve(
    chosen_logps: UnsafePointer[Float32, MutExternalOrigin],
    rejected_logps: UnsafePointer[Float32, MutExternalOrigin],
    chosen_ref_logps: UnsafePointer[Float32, MutExternalOrigin],
    rejected_ref_logps: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    beta: Float32,
    B: Int32,
) raises:
    pass
