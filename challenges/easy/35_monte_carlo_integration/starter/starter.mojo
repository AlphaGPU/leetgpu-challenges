from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# y_samples, result are device pointers
@export
def solve(y_samples: UnsafePointer[Float32], result: UnsafePointer[Float32], a: Float32, b: Float32, n_samples: Int32):
    pass
