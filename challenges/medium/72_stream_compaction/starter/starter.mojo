from std.gpu.host import DeviceContext
from std.memory import UnsafePointer

# A, out are device pointers
@export
fn solve(A: UnsafePointer[Float32, MutExternalOrigin], N: Int32, out: UnsafePointer[Float32, MutExternalOrigin]) raises:
    pass
