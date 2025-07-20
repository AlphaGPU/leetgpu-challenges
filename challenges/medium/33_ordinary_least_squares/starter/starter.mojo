from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# X, y, beta are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(X: UnsafePointer[Float32], y: UnsafePointer[Float32], beta: UnsafePointer[Float32], n_samples: Int32, n_features: Int32):
    pass
