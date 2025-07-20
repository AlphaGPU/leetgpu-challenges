from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(predictions: UnsafePointer[Float32], targets: UnsafePointer[Float32], mse: UnsafePointer[Float32], N: Int32):
    pass 