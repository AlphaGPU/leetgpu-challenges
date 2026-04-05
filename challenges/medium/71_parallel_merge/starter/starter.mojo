from std.gpu.host import DeviceContext
from std.memory import UnsafePointer

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], B: UnsafePointer[Float32, MutExternalOrigin], C: UnsafePointer[Float32, MutExternalOrigin], M: Int32, N: Int32) raises:
    pass
