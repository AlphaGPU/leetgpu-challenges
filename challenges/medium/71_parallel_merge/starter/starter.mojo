from gpu.host import DeviceContext
from memory import UnsafePointer

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M: Int32, N: Int32):
    pass
