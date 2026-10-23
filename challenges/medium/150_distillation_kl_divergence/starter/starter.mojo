from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# teacher_logits, student_logits, kl are device pointers
@export
def solve(
    teacher_logits: UnsafePointer[Float32, MutAnyOrigin],
    student_logits: UnsafePointer[Float32, MutAnyOrigin],
    kl: UnsafePointer[Float32, MutAnyOrigin],
    N: Int32,
    V: Int32,
    temperature: Float32,
) raises:
    pass
