from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# X, S, Y are device pointers
@export
def solve(X: UnsafePointer[Float32], S: UnsafePointer[Float32], Y: UnsafePointer[Float32], TILE_SIZE: Int32):
    pass
