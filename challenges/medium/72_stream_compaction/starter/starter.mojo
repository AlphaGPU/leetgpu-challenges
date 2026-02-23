from gpu.host import DeviceContext
from memory import UnsafePointer

# A, out are device pointers
@export
def solve(A: UnsafePointer[Float32], N: Int32, out: UnsafePointer[Float32]):
    pass
