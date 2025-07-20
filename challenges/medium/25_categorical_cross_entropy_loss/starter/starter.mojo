from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(logits: UnsafePointer[Float32], true_labels: UnsafePointer[Int32], loss: UnsafePointer[Float32], N: Int32, C: Int32):
    pass 