from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(agents: UnsafePointer[Float32], agents_next: UnsafePointer[Float32], N: Int32):
    pass 