from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(positions: UnsafePointer[Float32], masses: UnsafePointer[Float32], accelerations: UnsafePointer[Float32], N: Int32):
    pass
