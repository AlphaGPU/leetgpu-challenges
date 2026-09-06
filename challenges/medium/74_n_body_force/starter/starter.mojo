from gpu.host import DeviceContext
from memory import UnsafePointer

# positions, masses, forces are device pointers
@export
def solve(positions: UnsafePointer[Float32], masses: UnsafePointer[Float32], forces: UnsafePointer[Float32], N: Int32):
    pass
